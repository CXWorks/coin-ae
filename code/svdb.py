import svd2py
import sys


def parse_svd(svd_file:str):
    # Create SvdParser object passing path to SVD file
    parser = svd2py.SvdParser()
    # Invoke conver() function
    result = parser.convert(svd_file)
    return result


def walk_svd(ctxt: list, parent:str, unit):


    if isinstance(unit, dict):
        for k, v in unit.items():
            if k == 'enumeratedValues':
                if 'bitRange' in unit:
                    nums = unit['bitRange'][1:-1]
                    nums = nums.split(':')
                    ran = abs(int(nums[0]) - int(nums[1])) + 1
                    cases = 2**ran
                    print(unit['name'], nums, ran, cases, len(unit['enumeratedValues']['enumeratedValue']))
        for k, v in unit.items():
            ctxt.append(parent)
            walk_svd(ctxt, k, unit[k])
            ctxt.pop()

    elif isinstance(unit, list):
        for v in unit:
            ctxt.append(parent)
            walk_svd(ctxt, '$', v)
            ctxt.pop()


if __name__ == '__main__':
    svd = parse_svd(sys.argv[1])
    walk_svd([], '', svd)
