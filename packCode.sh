
wget https://github.com/danbarto/tW_scattering/archive/master.zip
unzip master.zip
mv tW_scattering-master tW_scattering

tar -czf tW_scattering.tar.gz tW_scattering

rm -rf tW_scattering
rm master.zip

mv tW_scattering.tar.gz Tools/analysis.tar.gz
