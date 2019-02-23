## Read output from output and create a csv file
## Check which iteration has the best config


import pandas as pd
import glob

if __name__ == '__main__':
    # list all the slurm outputs
    results = glob.glob('*.log')
    rows = []
    for res in results:
        r_file = open(res).readlines()
        for rf in r_file:
            if rf.startswith("> togrep :"):
                sp = rf.split(' : ')
                m_test_acc = float(sp[-1].lstrip())
                file_name = sp[1]
                rows.append({'file': file_name, 'test_acc':m_test_acc})
    df = pd.DataFrame(rows)
    df.to_csv('hyp_results.csv')