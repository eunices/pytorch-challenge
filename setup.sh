mkdir data
wget --directory-prefix=data/ https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip 
unzip data/*.zip -d data/
cd data && ls
