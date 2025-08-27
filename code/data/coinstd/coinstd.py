import csv
import json
import os
import zipfile
import warnings
import io
import datasets
import random
import numpy as np
import pickle
import pandas as pd
import subprocess
import lzma

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}

UNSAFELABELNAME = [
    "Safe",  # 0
    "CallToUnsafeFunction\n(internal)",  # 1
    "UseOfInlineAssembly",  # 2
    "InitializingTypeWith",  # 3
    "CastOfPointerToInt",  # 4
    "UseOfMutableStatic",  # 5
    "UseOfExternStatic",  # 6
    "DerefOfRawPointer",  # 7
    "AssignToDroppingUnionField",  # 8
    "AccessToUnionField",  # 9
    "MutationOfLayoutConstrainedField",  # 10
    "BorrowOfLayoutConstrainedField",  # 11
    "CallToFunctionWith",  # 12
    "UnsafeFunction",  # 13
    "CallToUnsafeFunction\n(external)",  # 14
]


def generate_prompt(text, st, ed, wst, wed):
    ct = 0
    text = text.splitlines(keepends=True)
    for i in range(st-1, ed):
        text[i] = '>\t'+text[i]
    t = st - 2
    while t >= 0:
        if text[t].strip().startswith('#') or text[t].strip().startswith('//'):
            text[t] = '>\t' + text[t]
            t -= 1
        else:
            break
    if wst == st and wed == ed:
        return ''.join(text), st, ed
    else:
        return ''.join(text[wst - 1: wed]), st, ed


def generate_prompt_unsafe(text, st, ed, wst, wed):
    text = text.splitlines(keepends=True)
    text[st-1] = text[st-1].replace('unsafe ','')
    for i in range(st-1, ed):
        text[i] = '>\t'+text[i]
    t = st-2
    while t>=0:
        if text[t].strip().startswith('#') or text[t].strip().startswith('//'):
            text[t] = '>\t'+text[t]
            t-=1
        else:
            break
    if wst == st and wed == ed:
        return ''.join(text), st, ed
    else:
        return ''.join(text[wst - 1: wed]), st, ed


def is_hex(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


# def get_func_instructions(fn_bin_meta, addr_start, addr_end, call_depth, max_insts, visited={}, addr2name=None):

#     fn_bin = fn_bin_meta['fn_bin']
#     zf = fn_bin_meta['zf']
#     addr2name = fn_bin_meta['addr2name']

#     root = os.path.dirname(fn_bin)
#     fn_func = f'{os.path.basename(fn_bin)}:{addr_start}:{addr_end}.bin'
#     visited.update({addr_start: None})

#     # handle function calls
#     n_insts = 0
#     code_ret = ""
#     assert fn_func in zf.namelist(), f"a file does not exist: {fn_func}"
#     for c in io.TextIOWrapper(zf.open(fn_func), "utf-8"):
#         c = c.strip()
#         code_ret += c + '\n'
#         n_insts += 1
#         if n_insts > max_insts:
#             break

#         # function call
#         mnemonic, op_str = c.split(',')[0], ','.join(c.split(',')[1:])

#         if 'call' in mnemonic and is_hex(op_str) and call_depth > 0 and (op_str not in visited) and max_insts - n_insts > 0:
#             fn_callee = addr2name[op_str]
#             fn_bin_callee, addr_start_callee, addr_end_callee = os.path.splitext(fn_callee)[0].split(":")
#             code_callee, n_insts_callee = get_func_instructions(fn_bin_meta, addr_start_callee, addr_end_callee, call_depth=call_depth-1, max_insts=max_insts - n_insts, visited=visited)
#             code_callee_post = "".join(['|<C>|' + l for l in io.StringIO(code_callee).readlines()])
#             code_ret += code_callee_post
#             n_insts += n_insts_callee

#     return code_ret, n_insts


def get_func_instructions(fn_bin_meta, addr_start, addr_end, call_depth, max_insts, visited={}, addr2name=None):
    fn_bin = fn_bin_meta['fn_bin']
    zf = fn_bin_meta['zf']
    addr2name = fn_bin_meta['addr2name']

    root = os.path.dirname(fn_bin)
    fn_func = f'{os.path.basename(fn_bin)}:{addr_start}:{addr_end}.bin'
    visited.update({addr_start: None})

    # handle function calls
    n_insts = 0
    code_caller_ret = ""
    code_callee_ret = []

    assert fn_func in zf.namelist(), f"a file does not exist: {fn_func}"
    for c in io.TextIOWrapper(zf.open(fn_func), "utf-8"):
        c = c.strip()
        code_caller_ret += c + '\n'
        n_insts += 1
        if n_insts > max_insts:
            break

        # function call
        mnemonic, op_str = c.split(',')[0], ','.join(c.split(',')[1:])

        # depth-first search for functions
        if 'call' in mnemonic and is_hex(op_str) and call_depth > 0 and (
                op_str not in visited) and max_insts - n_insts > 0:
            fn_callee = addr2name[op_str]
            fn_bin_callee, addr_start_callee, addr_end_callee = os.path.splitext(fn_callee)[0].split(":")
            code_callee, code_callee_callee, n_insts_callee = get_func_instructions(fn_bin_meta, addr_start_callee,
                                                                                    addr_end_callee,
                                                                                    call_depth=call_depth - 1,
                                                                                    max_insts=max_insts - n_insts,
                                                                                    visited=visited)

            code_callee_ret.append(code_callee)
            code_callee_ret += code_callee_callee

            n_insts += n_insts_callee

    return code_caller_ret, code_callee_ret, n_insts


def parse_objdump(f: str):
    with lzma.open(f, 'rb') as fp:
        outputb = fp.read()
        output = outputb.decode('utf-8')
        ans = {}
        infunc = False
        m = []
        fname = ''
        for l in output.splitlines(keepends=True):
            if l.startswith("000000"):
                if infunc:
                    ans[fname] = ''.join(m[:-1])
                    m = []
                else:
                    infunc = True
                    m = []
                fname = l[l.find('<') + 1:l.find('>')]
            elif l.startswith(' '):
                if infunc:
                    if '<' in l:
                        l = l[:l.find('<')] + '\n'
                    if '//' in l:
                        l = l[:l.find('//')] + '\n'
                    if '#' in l:
                        l = l[:l.find('#')] + '\n'

                    # once = False
                    # while idx < len(l):
                    #     if l[idx] in '1234567890qazwsxedcrfvtgbyhnujmikolp':
                    #         break
                    #     idx += 1
                    # idx += 8
                    # while idx < len(l):
                    #     if l[idx] in '1234567890qazwsxedcrfvtgbyhnujmikolp':
                    #         break
                    #     idx += 1
                    # idx = l.find(':') + 1
                    # addr = int(l[:idx])
                    if not l.endswith('\n'):
                        l = l + '\n'
                    ls = ' '.join([x for x in filter(lambda k: len(k) > 0, l[32:].split('\t'))])
                    # print(ls)
                    tmp = [x for x in filter(lambda k: len(k.strip()) > 0, ls.split(' '))]
                    # print(tmp)
                    if len(tmp) > 1:
                        tmp.insert(1, ',')
                    if len(tmp) > 0:
                        if not tmp[-1].endswith('\n'):
                            tmp[-1] += '\n'
                    ls = ''.join(tmp)
                    m.append(ls)
        if fname != '':
            ans[fname] = ''.join(m)
        return ans


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class Coin(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="caller", version=VERSION, data_dir='data/x64/crate_data',
                               description="crate dataset"),
        datasets.BuilderConfig(name="caller-sampled", version=VERSION, data_dir='data/x64/crate_data',
                               description="crate dataset"),
        datasets.BuilderConfig(name="caller-callee-ave", version=VERSION, data_dir='data/x64/crate_data',
                               description="crate dataset with callee code"),
    ]

    DEFAULT_CONFIG_NAME = "coin"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        # if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option1": datasets.Value("string"),
        #             "answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        # return datasets.DatasetInfo(
        #     # This is the description that will appear on the datasets page.
        #     description=_DESCRIPTION,
        #     # This defines the different columns of the dataset and their types
        #     features=features,  # Here we define them above because they are different between the two configurations
        #     # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
        #     # specify them. They'll be used if as_supervised=True in builder.as_dataset.
        #     # supervised_keys=("sentence", "label"),
        #     # Homepage of the dataset for documentation
        #     homepage=_HOMEPAGE,
        #     # License for the dataset if available
        #     license=_LICENSE,
        #     # Citation for the dataset
        #     citation=_CITATION,
        # )

        features = datasets.Features(
            {
                'function_text': datasets.Value("string"),
                'file_location': datasets.Value("string"),
                'label': datasets.features.ClassLabel(names=['safe', 'unsafe']),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)

        self.data_subdir = f'crate_data'

        postfix = ''
        if 'sampled' in self.config.name:
            postfix += '_sampled'

        return [
            #datasets.SplitGenerator(
            #    name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
            #    gen_kwargs={
            #        "filepath": 'train.pickle',
            #        'all_pickle': 'rand.pickle',
            #        "split": "train",
            #    },
            #),
            #datasets.SplitGenerator(
            #    name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
            #    gen_kwargs={
            #        "filepath": 'valid.pickle',
            #        'all_pickle': 'rand.pickle',
            #        "split": "valid",
            #    },
            #),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": 'test.pickle',
                    'all_pickle': 'rand.pickle',
                    "split": "test_std"
                },
            ),
        ]

    # def _generate_examples_caller(filepath, split, max_insts_per_func=500):
    #     call_depth = 0
    #     key = -1
    #     meta_info = {}

    #     with open(os.path.join(self.config.data_dir, filepath), encoding="utf-8") as f:
    #         for label in f.read().splitlines()[1:]:

    #             label_split = label.split(',')
    #             fn_bin = os.path.join(self.config.data_dir, self.data_subdir, 'bin', label_split[0])

    #             if not os.path.exists(fn_bin+'.bin.zip'):
    #                 warnings.warn(f'A file does not exit: {fn_bin}')
    #                 continue
    #             addr = label_split[1]
    #             addr_start = addr.split(':')[0]
    #             addr_end = addr.split(':')[1]
    #             unsafe_label = [UNSAFELABELNAME[int(e)] for e in label_split[2].split('/')]
    #             bug_label = [int(label_split[3])]
    #             current_program = int(label_split[4])

    #             key = key + 1

    #             try:

    #                 if fn_bin not in meta_info:
    #                     zf = zipfile.ZipFile(fn_bin + '.bin.zip', mode='r', compression=zipfile.ZIP_DEFLATED)
    #                     addr2name = {fn_callee.split(':')[1]: fn_callee for fn_callee in zf.namelist()}
    #                     meta_info[fn_bin] = {'fn_bin': fn_bin, 'zf': zf, 'addr2name': addr2name}

    #                 fn_bin_meta = meta_info[fn_bin]

    #                 function_asm, n_insts = get_func_instructions(fn_bin_meta, addr_start, addr_end, call_depth=call_depth, max_insts=max_insts_per_func, visited={})
    #             except OSError as e:
    #                 print(e)
    #                 print(fn_bin)
    #                 continue

    #             yield key, {"function_asm_text": function_asm, 'unsafe_label': unsafe_label}

    def _generate_examples_caller_callee(self, filepath, all, split, call_depth=3, max_insts_per_func=500):
        id = -1
        with open(f'/mnt/ssd1/xiang/coin3/coin_{split}.pkl', 'rb') as fp:
            data = pickle.load(fp)
            for k, vv in data.items():
                if k == 'safe':
                    for f, text, ls, window in vv:
                        #assert len(ls) == len(window)
                        #print(ls, window)
                        for idx, (st, ed) in enumerate(ls):
                            id+=1
                            prompt, _, _ = generate_prompt(text, st, ed, window[idx][0], window[idx][1])
                            yield id, {"file_location": f+f':{st}:{ed}', "function_text": prompt, 'label': 0}
                else:
                    for f, text, ls, window in vv:
                        for idx, (st, ed) in enumerate(ls):
                            id += 1
                            prompt, _, _ = generate_prompt_unsafe(text, st, ed, window[idx][0], window[idx][1])
                            yield id, {"file_location": f+f':{st}:{ed}', "function_text": prompt, 'label': 1}




        # max_insts = max_insts_per_func + call_depth * (max_insts_per_func * 5)
        # key = -1
        # meta_info = {}
        # all_data = {}
        # picked = {}
        # max_num_samples = {'train': 500_000_000, 'val': 200_000, 'test': 100_000_000}
        # with open(os.path.join(self.config.data_dir, all), 'rb') as fp:
        #     all_data = pickle.load(fp)
        # with open(os.path.join(self.config.data_dir, filepath), 'rb') as fp:
        #     picked = pickle.load(fp)
        # combined = {}
        # for (csv, funcname) in picked:
        #     if csv not in combined:
        #         combined[csv] = []
        #     if funcname not in combined[csv]:
        #         combined[csv].append(funcname)
        # for csv, vs in combined.items():
        #     data = csv.replace('.label.', '.data.')
        #     if os.path.exists(data):
        #         with lzma.open(data, 'rb') as fp:
        #             obj = fp.read()
        #             labels = pickle.loads(obj)
        #             for k, v in labels.items():
        #                 key += 1
        #                 yield key, {"function_asm_text": [v[0]], 'unsafe_label': v[1]}

        # objdump = csv.replace('.label.', '.txt.')
        # fnames = parse_objdump(objdump)
        # labels = None
        # with lzma.open(csv, 'rb') as fp:
        #     obj = fp.read()
        #     labels = pickle.loads(obj)
        # if labels is None:
        #     continue
        # qc = {}
        # for v in labels:
        #     qc[v['funcname']] = v['labels']
        # for v in vs:
        #     if v not in qc:
        #         print('err fn', v, csv, objdump)
        #         exit(-1)
        #     key += 1
        #     yield key, {"function_asm_text": fnames[v], 'unsafe_label': qc[v]}

        # for csv_file in picked:
        #     assert csv_file in all_data
        #     binary = csv_file[:-9]
        #     funcs = all_data[csv_file]
        #     df = pd.read_csv(os.path.join(self.config.data_dir, csv_file))
        #     if not os.path.exists(os.path.join('/mnt/sdc3/xiang/xz/', binary+'.xz')):
        #         print('missing', os.path.join('/mnt/sdc3/xiang/xz/', binary+'.xz'))
        #         continue
        #     asms = parse_objdump(os.path.join('/mnt/sdc3/xiang/xz/', binary))
        #     for func_idx in funcs:
        #         if key >= max_num_samples[split]:
        #             return
        #         key = key + 1
        #         row = df.loc[func_idx]
        #         asm = asms[func_idx]
        #         unsafe = str(row['unsafe'])
        #         labels = []
        #         if unsafe != 'nan':
        #             lset = set()
        #             for l in unsafe.split('-'):
        #                 lset.add(l)
        #             for x in lset:
        #                 try:
        #                     labels.append(int(x))
        #                 except:
        #                     labels.append(int(float(x)))
        #         yield key, {"function_asm_text": [asm], 'unsafe_label': labels}

        # with open(os.path.join(self.config.data_dir, filepath), encoding="utf-8") as f:
        #     for label in f.read().splitlines()[1:]:
        #
        #         label_split = label.split(',')
        #         fn_bin = os.path.join(self.config.data_dir, self.data_subdir, 'bin', label_split[0])
        #
        #         if not os.path.exists(fn_bin+'.bin.zip'):
        #             warnings.warn(f'A file does not exit: {fn_bin}')
        #             continue
        #         addr = label_split[1]
        #         addr_start = addr.split(':')[0]
        #         addr_end = addr.split(':')[1]
        #         unsafe_label = [UNSAFELABELNAME[int(e)] for e in label_split[2].split('/')]
        #         bug_label = [int(label_split[3])]
        #         current_program = int(label_split[4])
        #
        #         key = key + 1
        #
        #         try:
        #
        #             if fn_bin not in meta_info:
        #                 meta_info = {} # release memory
        #                 zf = zipfile.ZipFile(fn_bin + '.bin.zip', mode='r', compression=zipfile.ZIP_DEFLATED)
        #                 addr2name = {fn_callee.split(':')[1]: fn_callee for fn_callee in zf.namelist()}
        #                 meta_info[fn_bin] = {'fn_bin': fn_bin, 'zf': zf, 'addr2name': addr2name}
        #
        #             fn_bin_meta = meta_info[fn_bin]
        #
        #             function_asm_caller, function_asm_callee, n_insts = get_func_instructions(fn_bin_meta, addr_start, addr_end, call_depth=call_depth, max_insts=max_insts, visited={})
        #         except OSError as e:
        #             #print(e)
        #             warnings.warn(f'[{e}] {fn_bin}')
        #             continue
        #
        #         except zipfile.BadZipFile as e:
        #             #print(e)
        #             warnings.warn(f'[{e}] {fn_bin}')
        #             continue
        #
        #
        #         yield key, {"function_asm_text": [function_asm_caller] + function_asm_callee, 'unsafe_label': unsafe_label}

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, all_pickle, split):

        if self.config.name in ['caller', 'caller-sampled']:
            # return only code for a caller
            yield from self._generate_examples_caller_callee(filepath, all_pickle, split, call_depth=0,
                                                             max_insts_per_func=500)
        elif self.config.name in ['caller-callee-ave', 'caller-callee-ave-sampled']:
            # reeturn code for a caller along with the list of code of callees
            yield from self._generate_examples_caller_callee(filepath, all_pickle, split, call_depth=3,
                                                             max_insts_per_func=500)
        else:
            raise NotImplementedError
