import sys, pandas as pd
import produce_report as pr

def report(source, dataframe, task):
    if task == "classification":
        pr.classification(dataframe)
    elif task == "unsupervised":
        pr.unsupervised(dataframe)
    else:
        pr.preprocessing(dataframe)
         

if __name__ == '__main__':

    '''A: read arguments'''
    args = sys.stdin.readline().rstrip('\n').split(' ')
    n, source, task = int(args[0]), args[1], args[2]
    
    '''B: read dataset'''
    data, header = [], sys.stdin.readline().rstrip('\n').split(',')
    for i in range(n-1):
        data.append(sys.stdin.readline().rstrip('\n').split(','))    
    dataframe = pd.DataFrame(data, columns=header)

    '''C: output results'''
    print(report(source, dataframe, task))