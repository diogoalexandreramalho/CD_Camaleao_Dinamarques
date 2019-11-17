
def print_cnf_mtx(mtx, line):
    print("{}\t\t       Predicted".format(line))
    line += 1
    print("{}\t\t         N   P".format(line))
    line += 1
    print("{}\t\tTrue  N {}  {}".format(line, mtx[0,0], mtx[0,1]))
    line += 1
    print("{}\t\t      P {}  {}".format(line, mtx[1,0], mtx[1,1]))
    line += 1
    return line


def print_parameters(classifier, parameters):
    
    if classifier == "Naive Bayes":
        return parameters
    if classifier == "kNN":
        return "distance function = {}; nr neighbors = {}".format(parameters[0], parameters[1])
    if classifier == "Decision Tree":
        return "criteria = {}; max depths = {}; min samples leaf = {}".format(parameters[0], parameters[1], parameters[2])
    if classifier == "Random Forest":
        return "max features = {}; max depths = {}; nr estimators = {}".format(parameters[0], parameters[1], parameters[2])
    if classifier == "Gradient Boost":
        return "max features = {}; max depths = {}; nr estimators = {}; learning rate = {}".format(parameters[0], parameters[1], parameters[2], parameters[3])
    if classifier == "XGBoost":
        return "max depths = {}; nr estimators = {}".format(parameters[0], parameters[1])
    
def print_pre_parameters(pre_parameters):
    balanced = pre_parameters[0]
    normalized = pre_parameters[1]

    if balanced:
        params = "balanced "
    else:
        params = "unbalanced "
    if normalized:
        params += "and normalized"
    else:
        params += "and not normalized"
    return params

def print_stats(reports, pre_parameters):
    line = 1
    pre_process_params = print_pre_parameters(pre_parameters)
    print("{}1. Applied preprocessing: {}\n".format(line, pre_process_params))
    line+=1
    print("{}2. Classifiers:".format(line))
    line+=1
    for i in range(len(reports)):
        print("{}2.{} {}".format(line, i+1, reports[i][0]))
        line+=1
        print("{}\t2.{}.1 Best accuracy".format(line, i+1))
        line+=1
        best_accuracy = reports[i][1]
        accuracy_parameters = print_parameters(reports[i][0], best_accuracy[0])
        print("{}\ta) Suggested parameterization: {}".format(line, accuracy_parameters))
        line+=1
        print("{}\tb) Confusion matrix: ".format(line))
        line+=1
        line = print_cnf_mtx(best_accuracy[3], line)
        print("{}\t2.{}.2 Best sensitivity".format(line, i+1))
        line+=1
        best_sensitivity = reports[i][2]
        sensitivity_parameters = print_parameters(reports[i][0], best_sensitivity[0])
        print("{}\ta) Suggested parameterization: {}".format(line, sensitivity_parameters))
        line+=1
        print("{}\tb) Confusion matrix: ".format(line))
        line+=1
        line = print_cnf_mtx(best_sensitivity[3], line)
        print(line)
        line+=1
    print("{}3. Comparative performance: NB | kNN | DT | RF | GB | XGB".format(line))
    line+=1
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format("{0:.2f}".format(report[1][1]))
    print("{}\t3.1 Accuracy: ".format(line) + accuracies[:-2])
    line+=1
    sensitivities = ""
    for report in reports:
        sensitivities += "{} | ".format("{0:.2f}".format(report[2][2]))
    print("{}\t3.2 Sensitivity: ".format(line) + sensitivities[:-2])
    line+=1
        
