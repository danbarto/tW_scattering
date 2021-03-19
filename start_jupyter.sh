
echo "Starting up jupyter server. Once this is done, run the following command on your computer:"
echo "  ssh -N -f -L localhost:8893:localhost:8893 $USER@uaf-10.t2.ucsd.edu"

jupyter nbextension enable widgetsnbextension --user --py
jupyter notebook --no-browser --port=8893

