
source common.ini
#codepath=/Users/xirong/bitbucket/im2concept
#rootpath=/Users/xirong/VisualSearch

#./download_voc_data.sh

trainCollection=voc2008train
trainAnnotationName='conceptsvoc2008train.txt'

valCollection='voc2008val'
valAnnotationName='conceptsvoc2008val.txt'

testCollection='voc2008val'
testAnnotationName=$valAnnotationName

feature=dsift

minmaxfile=$rootpath/$trainCollection/FeatureData/$feature/minmax.txt
if [ ! -f "$minmaxfile" ]; then
    feat_dir=$rootpath/$trainCollection/FeatureData/$feature
    python fiksvm/find_min_max.py $feat_dir
fi


for modelName in fastlinear fik
do
    echo "optimize hyper parameters for $modelName"
    python optimize_hyper_params.py $trainCollection $trainAnnotationName $valCollection $valAnnotationName $feature $modelName --rootpath $rootpath
done

python fastlinear/trainLinearConcepts.py $trainCollection $trainAnnotationName $feature --rootpath $rootpath

best_param_dir=$rootpath/$trainCollection/Models/$trainAnnotationName/fastlinear,best_params/$valCollection,$valAnnotationName,$feature

python fastlinear/trainLinearConcepts.py $trainCollection $trainAnnotationName $feature --best_param_dir $best_param_dir --rootpath $rootpath


python fiksvm/trainFikConcepts.py $trainCollection $trainAnnotationName $feature --rootpath $rootpath

best_param_dir=$rootpath/$trainCollection/Models/$trainAnnotationName/fik50,best_params/$valCollection,$valAnnotationName,$feature

python fiksvm/trainFikConcepts.py $trainCollection $trainAnnotationName $feature --best_param_dir $best_param_dir --rootpath $rootpath


for modelName in fastlinear fastlinear-tuned fik50 fik50-tuned
do
    modelfile=$rootpath/$trainCollection/Models/$trainAnnotationName/$feature/$modelName/dog.model
    if [ ! -f "$modelfile" ]; then
        echo "$modelfile does not exist!"
        continue
    fi
    python applyConcepts.py $testCollection $trainCollection $trainAnnotationName $feature $modelName --rootpath $rootpath
done


#python ~/myCode/cross-platform/autotagging/evalTagvotes.py $testCollection $testAnnotationName $codepath/im2concept/runs-$testCollection.txt

