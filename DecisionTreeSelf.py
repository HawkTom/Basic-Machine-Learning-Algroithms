import RegressionTree as rt

dataFile = "train.txt"
output_file_dot = "regression.dot"
output_file_pdf = "regression.pdf"

data = rt.dataRead(dataFile)  # output the train data from the file
x = rt.createTree(data) # create the regression tree by the data
rt.dot_File(x, output_file_dot) # output the tree information to dot file
model = rt.plot_model(data) # plot the line and regression tree model in th graph
print(model)