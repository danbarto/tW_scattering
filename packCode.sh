
#wget https://github.com/danbarto/tW_scattering/archive/master.zip
#unzip master.zip
#mv tW_scattering-master tW_scattering

mkdir -p tW_scattering/Tools/
mkdir -p tW_scattering/data/
cp -r Tools/*.py tW_scattering/Tools/
cp -r data/ tW_scattering/

tar -czf tW_scattering.tar.gz tW_scattering

rm -rf tW_scattering

mv tW_scattering.tar.gz Tools/analysis.tar.gz
