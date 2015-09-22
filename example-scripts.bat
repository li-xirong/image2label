

set rootpath=e:\xirong\VisualSearch
set trainCollection=voc2008train
set trainAnnotationName=conceptsvoc2008train.txt
set feature=dsift

set minmaxfile=%rootpath%\%trainCollection%\FeatureData\%feature%\minmax.txt
set featdir=%rootpath%\%trainCollection%\FeatureData\%feature%
IF EXIST %minmaxfile% (
    echo %minmaxfile% exists
) ELSE (
    python fiksvm/find_min_max.py %featdir%
)

python fastlinear/trainLinearConcepts.py %trainCollection% %trainAnnotationName% %feature% --rootpath %rootpath%
python fiksvm/trainFikConcepts.py        %trainCollection% %trainAnnotationName% %feature% --rootpath %rootpath%
