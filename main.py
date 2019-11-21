import sys
import pandas as pd
import produce_report as pr
import Unsupervised as unsup

def report(source, dataframe, task):
    task = task[:-1]
    
    dataframe.insert(755, 'class', 0)
    i = 0
    for value in dataframe['class\r']:
        new_value = int(value[:-1])
        dataframe.loc[i, 'class'] = new_value
        i+=1
    del dataframe['class\r']
    
    if task == "classification":
        pr.classification(dataframe)
    elif task == "unsupervised":
        unsup.run(source, dataframe)
    else:
        #pr.preprocessing(dataframe)
        pass
         

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
    report(source, dataframe, task)