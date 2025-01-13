#! /bin/bash
BASE_DIR='data/accubrainresult'
IMAGE='T1_rigid.nii.gz'
LABEL='result_seg_rigid.nii.gz'
PYTHON_PATH=python

if which pipenv | grep -vq "not found"
then
	if pipenv --venv | grep -q "/"
	then
		PYTHON_PATH=`pipenv --venv`/bin/python
	fi
fi

echo "Use pipenv $PYTHON_PATH"

for dir in `ls -R data/accubrainresult/ | grep : | sed 's/://'`
do
	if ls $dir | grep -q $IMAGE
	then
		echo $dir
		$PYTHON_PATH main.py -i $dir/$IMAGE -l $dir/$LABEL -o $dir -d data.txt both
	fi
done	
