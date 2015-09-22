
The **image2label** package implements SVM based image concept recognition, with

* a high-level python interface for loading labels, feature vectors, and SVM models, training and applying models
* linear SVMs and fast intersection kernel SVMs
* model compression that allows new models to be added into an existing model
* cross-platform support (linux, mac, windows) 

It has been used in multiple tasks including visual categorization [1], tag relevance learning [2], and object localization [3].

##Prerequisites

* The package does not include any visual feature extractors. Features of training and test data need to be pre-computed, and converted to required binary format using [txt2bin.py](https://github.com/li-xirong/simpleknn/blob/master/txt2bin.py).
* To minimize one's coding effort, the package requires training data and test data to be organized in a fixed structure, see the [sample data](http://www.mmc.ruc.edu.cn/research/negbp/voc2008train-voc2008val.zip).

##Getting started

####Setup

* Add [simpleknn](https://github.com/li-xirong/simpleknn) and `image2label` to `PYTHONPATH`
* Change `ROOT_PATH` in [basic/constant.py](basic/constant.py) to local folder where training and test data are stored
* Download sample data (from PASCAL VOC 2008), using the script [download\_voc\_data.sh](download_voc_data.sh) or via [google drive](https://drive.google.com/file/d/0B89Vll9z5OVEMk1oNTRaSXE5NUE/view?usp=sharing)
* Check if everything can be properly loaded by running [predict\_and\_eval.py](predict_and_eval.py) 

The following code samples show how to learn a linear classifier for a given label, say *dog*, from `voc2008train`, and evaluate the classifier on `voc2008val`, with Average Precision (AP) as the performance metric.


####Step 1. Load annotations

	from basic.constant import ROOT_PATH
	from basic.annotationtable import readAnnotationsFrom
        
	rootpath = ROOT_PATH
	trainCollection = 'voc2008train'
	trainAnnotationName = 'conceptsvoc2008train.txt'
	concept = 'dog'
	names,labels = readAnnotationsFrom(trainCollection, trainAnnotationName, concept, skip_0=True, rootpath=rootpath)
	name2label = dict(zip(names,labels))


####Step 2. Load feature vectors

	from simpleknn.bigfile import BigFile

	feature = 'dsift'
	train_feat_file = BigFile(os.path.join(rootpath,trainCollection,'FeatureData',feature))
	renamed,vectors = train_feat_file.read(names)
	Ys = [name2label[x] for x in renamed]


####Step 3. Train model

	from fastlinear.liblinear193.python.liblinearutil import train as train_model

	svm_params += ' -s 2 -B -1 '
	model = train_model(Ys, vectors, svm_params + ' -q')

####Step 4. Compress trained model for fast prediction
	
	from fastlinear.fastlinear import liblinear_to_fastlinear as compress_model

	feat_dim = train_feat_file.ndims
	new_model = compress_model([model], [1.0], feat_dim, params={})


####Step 5. Prediction

	testCollection = 'voc2008val'
	testAnnotationName = 'conceptsvoc2008val.txt'

	from basic.util import readImageSet
	test_imset = readImageSet(testCollection, testCollection, rootpath=rootpath)
	test_feat_file = BigFile(os.path.join(rootpath,testCollection,'FeatureData',feature))
	test_renamed, test_vectors = test_feat_file.read(test_imset)
	ranklist = [(test_renamed[i], new_model.predict(test_vectors[i])) for i in range(len(test_renamed))]
	ranklist.sort(key=lambda v:v[1], reverse=True)


####Step 6. Evaluation
	
	names,labels = readAnnotationsFrom(testCollection, testAnnotationName, concept, skip_0=True, rootpath=rootpath)
	test_name2label = dict(zip(names,labels))
	sorted_labels = [test_name2label[x[0]] for x in ranklist]

	from basic.metric import getScorer
	scorer = getScorer('AP')
	perf = scorer.score(sorted_labels)

See [predict\_and\_eval.py](predict_and_eval.py) for a complete code to compute AP scores for an array of concept models.        

####Save model to file
	# linear SVMs
	from fastlinear.fastlinear import fastlinear_save_model as save_model

	# fik SVMs
	from fiksvm.fiksvm import fiksvm_save_model as save_model
	
	save_model(model_file_name, new_model)

####Load model from file
	
	# linear SVMs
	from fastlinear.fastlinear import fastlinear_load_model as load_model

	# fik SVMs
	from fiksvm.fiksvm import fiksvm_load_model as load_model
	
	model = load_model(model_file_name)    

####Add a new (compressed) model to an existing model

	# model_1 <- model_1 * w1 + model_2 * w2
	model_1.add_fastsvm(model_2, w1, w2)



### References

[1] Xirong Li, Cees G. M. Snoek, Marcel Worring, Dennis Koelma, Arnold W. M. Smeulders, Bootstrapping visual categorization with relevant negatives, IEEE Transactions on Multimedia, 15(4):933-945, 2013

[2] Xirong Li, Cees G. M. Snoek,  Classifying tag relevance with relevant positive and negative examples, ACM Multimedia, 2013

[3] Xirong Li, Qin Jin, Shuai Liao, Junwei Liang, Xixi He, Yujia Huo, Weiyu Lan, Bin Xiao, Yanxiong Lu, Jieping Xu: RUC-Tencent at ImageCLEF 2015: Concept Detection, Localization and Sentence Generation, CLEF Working Notes 2015


