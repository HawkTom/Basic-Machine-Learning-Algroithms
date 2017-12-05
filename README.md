### Content



***This repository is mainly for place the code written myself in machine leaning*** 



**File structure**

```go
|————Cluster
	├── gmm.py    		// gaussian mixture model
	├── img_pro.py  	//Image Compression by using kmeans
		└── panda.png   //example for image compression 
	└── K_mean.py       //Kmean an K-medoids algorithm

|————Decision tree
	├── Classification Tree.py    //Decision tree for classification without pruning
	├── Decision Tree Pruning.py  //Decision tree for classification with pruning
		└── lense.txt
	└── Regression Tree.py 		  //decision tree for regression
		├── train.txt
		└── test.txt

|————Neural Network
	lab8
    ├── data
    │   └── cifar-10-batches-py   // * Not provided. You need to download it from 	 
			                     // <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>
    │       ├── banches.meta
    │       ├── data_batch_1
    │       └── ...
    ├── cifar10.py                // Helper functions for loading data
    ├── lab8.py                   // You should finish the functions in this file
    └── main.py                   // Run this file to see the results

|————Support Vector Machine // it has not finished yet. 
	├── SVM_.py    // the main impelement of SVM, include Visualization and SMO algorithm
	├── SVM_test.py // generate a data sample to test SVM algorithm
		└── S.npy is its model data
	└── data file
		├── svm_data.txt // data for linear kernel 
		├── svm_data2.txt // data for gaussian kernel 
			└── SVM.npy is its model data
		└── svm_data_two_cycle.txt // data for gaussian kernel

└── others // some other codes like A* search algorithm and so on
```

