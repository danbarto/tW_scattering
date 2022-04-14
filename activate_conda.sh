export TWHOME=$PWD
export PYTHONPATH=${PYTHONPATH}:$PWD:$PWD/coffea/:$PWD/postProcessing/ProjectMetis/:$PWD/BIT/

#conda activate coffeadev4
conda activate workerenv

#( conda activate daskanalysisenv && jupyter notebook --no-browser )
