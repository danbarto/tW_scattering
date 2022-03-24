
#wget https://github.com/danbarto/tW_scattering/archive/master.zip
#unzip master.zip
#mv tW_scattering-master tW_scattering

mkdir -p tW_scattering/Tools/
mkdir -p tW_scattering/data/
mkdir -p tW_scattering/processor/
mkdir -p tW_scattering/ML/
cp -r Tools/*.py tW_scattering/Tools/
cp -r processor/*.py tW_scattering/processor/
cp -r data/ tW_scattering/
cp -r ML/*.py tW_scattering/ML/
cp -r ML/networks/ tW_scattering/ML/

tar -czf tW_scattering.tar.gz tW_scattering

rm -rf tW_scattering

mv tW_scattering.tar.gz Tools/analysis.tar.gz
