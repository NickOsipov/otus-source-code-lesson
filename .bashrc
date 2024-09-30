export PYTHONPATH=$(pwd)
export YC_ZONE=ru-central1-a
export YC_SUBNET_NAME=default-ru-central1-a
export YC_SA_NAME=otus
export YC_USER=ubuntu

function log() {
    echo $(date) "| INFO:" $@
}