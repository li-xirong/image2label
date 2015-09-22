
import os

from basic.constant import ROOT_PATH
from basic.util import readImageSet
from basic.annotationtable import readConcepts,readAnnotationsFrom
from basic.metric import getScorer
from simpleknn.bigfile import BigFile

if __name__ == '__main__':
    rootpath = ROOT_PATH
    trainCollection = 'voc2008train'
    trainAnnotationName = 'conceptsvoc2008train.txt'
    
    testCollection = 'voc2008val'
    testAnnotationName = 'conceptsvoc2008val.txt'

    feature = 'dsift'
    modelName = 'fastlinear'
    modelName = 'fik50'
    metric = 'AP'
    scorer = getScorer(metric)


    if modelName.startswith('fik'):
        from fiksvm.fiksvm import fiksvm_load_model as load_model
    else:
        from fastlinear.fastlinear import fastlinear_load_model as load_model

    test_imset = readImageSet(testCollection, testCollection, rootpath=rootpath)
    test_feat_file = BigFile(os.path.join(rootpath,testCollection,'FeatureData',feature))
    test_renamed, test_vectors = test_feat_file.read(test_imset)

    concepts = readConcepts(testCollection, testAnnotationName, rootpath=rootpath)

    print ('### %s' % os.path.join(trainCollection, 'Models', trainAnnotationName, feature, modelName))
    results = []

    for concept in concepts:
        model_file_name = os.path.join(rootpath, trainCollection, 'Models', trainAnnotationName, feature, modelName, '%s.model' % concept)
        model = load_model(model_file_name)

        ranklist = [(test_renamed[i], model.predict(test_vectors[i])) for i in range(len(test_renamed))]
        ranklist.sort(key=lambda v:v[1], reverse=True)

        names,labels = readAnnotationsFrom(testCollection, testAnnotationName, concept, skip_0=True, rootpath=rootpath)
        test_name2label = dict(zip(names,labels))
        sorted_labels = [test_name2label[x[0]] for x in ranklist if x[0] in test_name2label]
        perf = scorer.score(sorted_labels)

        print ('%s %g' % (concept, perf))

        results.append((concept, perf))

    mean_perf = sum([x[1] for x in results]) / len(concepts)
    print ('mean%s %g' % (metric, mean_perf))



