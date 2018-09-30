# play-me-something
Repository for the machine learning part of play-me-something.
All of this code is based on the work from my Master's thesis,
[Deep Neural Networks for Context Aware Personalized Music Recommendation](http://kth.diva-portal.org/smash/record.jsf?pid=diva2:1118011) conducted at Spotify.

## Setting up gcloud authentication
```bash
gcloud auth application-default login
```

## Setting up Tensorflow environment
Create a virtualenv. System site packages need to be included because Sparkey is only debian packaged.

```bash
virtualenv --system-site-packages env/
source env/bin/activate
pip install -r requirements.txt
```

## Setup with a Mac & Anaconda:
```
conda create -n env python=2.7
source activate env
pip install -r requirements.txt
```

## Training on CloudML
To train on CloudML, do something like:
```
JOBNAME=deeper_model_subset_1 && gcloud --project pms ml-engine jobs submit training $JOBNAME --package-path=pms --module-name=pms.run --staging-bucket=gs://play-me-something-thesis --region=us-central1 --config=cloudml.yaml -- $(./params.py  < pms/config/deeper_model_cloud.yaml) --train --cloud --checkpoint-dir=gs://play-me-something-thesis/$USER/jobs/$JOBNAME --runtime-version=1.0
```
or if already have a yaml file:
```
JOBNAME=deeper_model_subset_1 && gcloud --project pms ml-engine jobs submit training $JOBNAME --package-path=pms --module-name=pms.run --staging-bucket=gs://play-me-something-thesis --region=us-central1 --config=cloudml.yaml -- $(./params.py  < pms/config/deeper_model_cloud.yaml) --train --cloud --checkpoint-dir=gs://play-me-something-thesis/$USER/jobs/$JOBNAME --runtime-version=1.0
```

# Tensorboard
To show tensorboard for all pre-trained models run:
```bash
python trained/tensorboard.py
```
