import bisect
import multiprocessing
import os
import json
import multiprocessing as mp
import shlex
import pickle
import pandas as pd
import subprocess
import sys
import lzma
from elftools.elf.elffile import ELFFile


def decode_file_line(dwarfinfo):
    # Go over all the line programs in the DWARF information, looking for
    # one that describes the given address.
    ans = set()
    for CU in dwarfinfo.iter_CUs():
        # First, look at line programs to find the file/line for the address
        line_program = dwarfinfo.line_program_for_CU(CU)
        delta = 1 if line_program.header.version < 5 else 0
        prevstate = None
        for entry in line_program.get_entries():
            # We're interested in those entries where a new state is assigned
            if entry.state is None:
                continue
            # Looking for a range of addresses in two consecutive states that
            # contain the required address.
            if prevstate and prevstate.address < entry.state.address and prevstate.address != 0:
                file_entry = line_program['file_entry'][entry.state.file - delta]
                filename = file_entry.name
                directory = line_program['include_directory'][file_entry.dir_index - delta]
                line = prevstate.line
                ans.add((prevstate.address, entry.state.address, directory.decode('utf-8')+'/'+filename.decode('utf-8'), line))
            if entry.state.end_sequence:
                # For the state with `end_sequence`, `address` means the address
                # of the first byte after the target machine instruction
                # sequence and other information is meaningless. We clear
                # prevstate so that it's not used in the next iteration. Address
                # info is used in the above comparison to see if we need to use
                # the line information for the prevstate.
                prevstate = None
            else:
                prevstate = entry.state
    res = [x for x in ans]
    res.sort(key=lambda x:x[0])
    return res

def recur_scan(folder:str, ans: list):
    if not os.path.exists(folder):
        return
    for f in os.listdir(folder):
        path = folder +'/'+f
        if os.path.isfile(path) and os.access(path, os.X_OK) and not (path.endswith(".sh") or path.endswith(".py")) and check_file(path):
            if '/build/' not in path:
                ans.append(path)
        if os.path.isdir(path) and not os.path.islink(path):
            recur_scan(path, ans)


def check_file(f:str):
    fin = subprocess.run('readelf -h {}'.format(f), shell=True, capture_output=True)
    if fin.returncode == 0:
        output = fin.stdout.decode('utf-8')
        if 'X86-64' in output:
            return True
    return False

def parse_objdump(f : str):
    fin = subprocess.run("objdump -M intel -d {}".format(f), shell=True, capture_output=True)
    assert fin.returncode == 0
    output = fin.stdout.decode("utf-8")
    ans = []
    infunc = False
    m=None
    fnames = []
    addrs = {}
    for l in output.splitlines():
        if l.startswith("000000"):
            if infunc:
                m['addrs'] = addrs
                ans.append(m)
            st = l.find('<')
            ed = l.find('>')
            start = int(l[:st-1], 16)
            infunc = True
            addrs= {}
            addrs[start] = ''
            m = {'funcname': l[st+1:ed], 'start': start}
            fnames.append(m)
        else:
            if infunc:
                ed = l.find(':')
                if ed > 0:
                    try:
                        end = int(l[:ed], 16)
                        m['end'] = end
                        addrs[end]=''
                    except:
                        pass
    if infunc:
        m['addrs'] = addrs
        ans.append(m)
    # fnames.sort(key= lambda c: c['start'])
    return fnames


def addr2line_old(fnames:list, f:str):
    p = subprocess.Popen(shlex.split('/usr/bin/addr2line -e {}'.format(f)), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    inputs = []
    rtb = {}
    for idx, fn in enumerate(fnames):
        ks = [x for x in fn['addrs'].keys()]
        inputs.extend([hex(x) for x in ks])
        for x in ks:
            rtb[x] = idx
    (out,err) = p.communicate('\n'.join(inputs))
    outs = out.splitlines()
    for idx in range(len(inputs)):
        src = outs[idx]
        addr = int(inputs[idx], 16)
        fidx = rtb[addr]
        fnames[fidx]['addrs'][addr] = src


def bsearch(res:list, tar:int):
    end = bisect.bisect(res, tar, key=lambda x:x[0])
    for i in reversed(range(end)):
        st,ed, file, line = res[i]
        if st<=tar and ed > tar:
            return file, line
        if ed < tar:
            break
    return '/no_file', 0


def addr2line(fnames:list, f:str, root:str):
    with open(f, 'rb') as fp:
        elf = ELFFile(fp)
        code = elf.get_section_by_name('.text')
        dwarfinfo = elf.get_dwarf_info()
        a2l = decode_file_line(dwarfinfo)
        for idx, fn in enumerate(fnames):
            for x in fn['addrs'].keys():
                file, line = bsearch(a2l, x)
                if not file.startswith('/'):
                    file = root + file
                fn['addrs'][x] = '{}:{}'.format(file, line)



def parse_log(f:str, unsafe_info:dict):
    with open(f, 'r') as fp:
        ls = fp.readlines()
        file = ''
        count = 0
        inside = False
        for idx in range(len(ls)):
            l = ls[idx]
            if inside:
                count -= 1
                l = l.split('\t')[1]
                lw = l.split(' ')
                line = int(lw[0]) + 1
                label = int(lw[3])
                if file not in unsafe_info:
                    unsafe_info[file] = {}
                if line not in unsafe_info[file]:
                    unsafe_info[file][line] = set()
                unsafe_info[file][line].add(label)
                if count == 0:
                    inside = False

            if l.startswith('find_unsafe_at'):
                lw = l.split(' ')
                file = lw[1]
                count = int(lw[2])
                inside = True









def parse_exec(stderr:str, root:str):
    ans = []
    for l in stderr.splitlines():
        if 'Executable ' in l:
            file = l[l.find('(')+1:l.find(')')]
            if os.path.exists(root + '/' +file):
                ans.append(root + '/' +file)
    return ans


def download_build(args):
    crate = args[0]
    version = args[1]
    dst = args[2]
    binary = []
    url = "https://static.crates.io/crates/{}/{}-{}.crate".format(crate, crate, version)
    ret = subprocess.run("wget -q -t 3 {} && mv {}-{}.crate {}-{}.tar.gz && tar zxf {}-{}.tar.gz && rm {}-{}.tar.gz".format(url, crate, version, crate, version, crate, version, crate, version), shell=True, capture_output=True)
    if ret.returncode != 0:
        print("error in {}-{}".format(crate, version))
        print(ret.stderr.decode("utf-8"))
        return binary
    try:
        fin = subprocess.run("rustup show", cwd='{}-{}'.format(crate, version), shell=True, capture_output=True)
        default_check = False
        for l in fin.stdout.decode('utf-8').splitlines():
            if 'rust183' in l and '(default)' in l:
                default_check = True
        if not default_check:
            fin = subprocess.run("rustup override set rust183", cwd='{}-{}'.format(crate, version), shell=True, capture_output=True)
        ret = subprocess.run('cd {}-{} && cargo clean && export SHOW_UNSAFE=1 && cargo build -j 1'.format(crate, version)
                             , shell=True, capture_output=True, timeout=3600)
        if ret.returncode == 0:
            with open('{}-{}/compile_out.txt'.format(crate, version),'w') as fp:
                fp.write(ret.stdout.decode("utf-8"))
            # # parse
            # binary.extend(parse_exec(ret.stderr.decode('utf-8'),'{}-{}'.format(crate, version)))
            # ret2 = subprocess.run('cd {}-{} && export SHOW_UNSAFE=1 && export RUSTCFLAGS="-C opt-level=3" && cargo build -j 1 --examples'.format(crate, version)
            #                       , shell=True, capture_output=True, timeout=3600)
            # if ret2.returncode == 0:
            #     with open('{}-{}/compile_out.txt'.format(crate, version),'a') as fp:
            #         fp.write(ret2.stdout.decode("utf-8"))
            #     binary.extend(parse_exec(ret2.stderr.decode('utf-8'),'{}-{}'.format(crate, version)))
            #     recur_scan('{}-{}/target/debug/'.format(crate, version), binary)
            # else:
            #     with open('{}-{}/compile_test_err.txt'.format(crate, version),'w') as fp:
            #         fp.write(ret2.stderr.decode("utf-8"))
            # ret2 = subprocess.run('cd {}-{} && export SHOW_UNSAFE=1 && export RUSTCFLAGS="-C opt-level=3" && cargo test -j 1 --no-run'.format(crate, version)
            #                       , shell=True, capture_output=True, timeout=3600)
            # if ret2.returncode == 0:
            #     with open('{}-{}/compile_test_out.txt'.format(crate, version),'w') as fp:
            #         fp.write(ret2.stdout.decode("utf-8"))
            #     binary.extend(parse_exec(ret2.stderr.decode('utf-8'),'{}-{}'.format(crate, version)))
            # else:
            #     with open('{}-{}/compile_test_err.txt'.format(crate, version),'a') as fp:
            #         fp.write(ret2.stderr.decode("utf-8"))
            ret = subprocess.run(
                'cd {}-{} && cargo clean'.format(crate, version)
                , shell=True, capture_output=True, timeout=3600)
            ret = subprocess.run(
                'mv {}-{} {}'.format(crate, version, dst)
                , shell=True, capture_output=True, timeout=3600)
        else:
            print("error compile in {}-{}".format(crate, version))
            with open('{}-{}/compile_out.txt'.format(crate, version),'w') as fp:
                fp.write(ret.stdout.decode("utf-8"))
            with open('{}-{}/compile_err.txt'.format(crate, version),'w') as fp:
                fp.write(ret.stderr.decode("utf-8"))
            ret = subprocess.run(
                'rm -rf {}-{}'.format(crate, version)
                , shell=True, capture_output=True, timeout=3600)
        return binary
    except subprocess.TimeoutExpired as exc:
        print("timeout in {}-{}".format(crate, version))
        ret = subprocess.run(
            'rm -rf {}-{}'.format(crate, version)
            , shell=True, capture_output=True, timeout=3600)
        with open('{}-{}/compile_err.txt'.format(crate, version),'w') as fp:
            fp.write('{} timeout\n'.format(crate))
        return binary


def one_process(args):
    crate = args[0]
    version = args[1]
    try:
        binary = download_build(args)

    except Exception as e:
        print('err', crate, version, e)
    #     if len(binary) > 0:
    #         unsafe_info = {}
    #         parse_log('/mnt/md0/xiang/rust_repo/rust/stdlib.x64.txt', unsafe_info)
    #         if os.path.exists('{}-{}/compile_out.txt'.format(crate, version)):
    #             parse_log('{}-{}/compile_out.txt'.format(crate, version), unsafe_info)
    #         if os.path.exists('{}-{}/compile_test_out.txt'.format(crate, version)):
    #             parse_log('{}-{}/compile_test_out.txt'.format(crate, version), unsafe_info)
    #         for bina in binary:
    #             fnames = parse_objdump(bina)
    #             addr2line(fnames, bina, os.getcwd()+'/{}-{}/'.format(crate, version))
    #             found = False
    #             for fn in fnames:
    #                 labels = set()
    #                 addr2label = {}
    #                 for addr, fl in fn['addrs'].items():
    #                     file = fl.split(':')[0]
    #                     try:
    #                         line = int(fl.split(':')[1])
    #                         #print(file, file in unsafe_info)
    #                         if file in unsafe_info and line in unsafe_info[file]:
    #                             for lb in unsafe_info[file][line]:
    #                                 labels.add(lb)
    #                                 if addr not in addr2label:
    #                                     addr2label[addr] = []
    #                                 if lb not in addr2label[addr]:
    #                                     addr2label[addr].append(lb)
    #                     except:
    #                         pass
    #                 del fn['addrs']
    #                 #print(addr2label)
    #                 fn['addr2label'] = addr2label
    #                 fn['labels'] = [x for x in labels]
    #                 if len(addr2label) > 0:
    #                     print(fn)
    #                     found = True
    #             if found:
    #                 with open(bina+'.label', 'wb') as fp:
    #                     pickle.dump(fnames, fp)
    #                 transfer(bina+'.label', bina, dst)
    # except Exception as e:
    #     print('err',crate,version,e)
    #subprocess.run('rm -rf {}-{}'.format(crate, version), shell=True)


def transfer(label: str, binary:str, dst: str):
    folder = label[:label.rfind('/')]
    folder = dst+folder
    if not os.path.exists(folder):
        subprocess.run("mkdir -p {}".format(folder), shell=True)
    subprocess.run("mv {} {} && xz {}".format(label, folder, dst+label), shell=True)
    subprocess.run("objdump -d {} > {}.txt && mv {}.txt {} && xz {}.txt".format(label[:-6],label[:-6], label[:-6], folder, dst + label[:-6]), shell=True)


if __name__ == '__main__':
    # fnames = parse_objdump('../wasm')
    # addr2line(fnames, '../wasm')
    data_file = sys.argv[1]
    df = pd.read_csv(data_file)
    total = 0
    min_start = int(sys.argv[2])
    wl = []
    for idx, row in df.iterrows():
        if idx < min_start:
            continue
        if idx >= int(sys.argv[3]):
            break
        crate = str(row['name'])
        version = str(row['num'])
        wl.append((crate, version, sys.argv[4]))
        total+=1
    print(total)
    with multiprocessing.Pool(96) as pool:
        pool.map(one_process, wl)
    print(total)
