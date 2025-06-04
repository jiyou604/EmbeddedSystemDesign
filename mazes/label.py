import glob

label_files = glob.glob('./*/labels/*.txt')

for f in label_files:
    with open(f) as infile:
        lines = infile.readlines()
    # 0 아닌 줄은 모두 0으로 바꿔 저장 (단일 클래스일 때만!)
    new_lines = []
    for l in lines:
        s = l.strip().split()
        s[0] = '0'
        new_lines.append(' '.join(s)+'\n')
    with open(f, 'w') as outfile:
        outfile.writelines(new_lines)
