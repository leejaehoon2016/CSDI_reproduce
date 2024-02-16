#!/bin/bash
PROJECT=`basename $PWD`
WORKDIR="/home/$USER/workspace"
DATADIR="/mnt/disks/sdb/$USER/data"
GPU=${GPU:-all}

if [[ $1 == "jupyter" ]]; then
	NAME="${NAME:-$1}"
	OPTIONS="${OPTIONS} -p $PORT:8888"
elif [[ $1 == "tensorboard" ]]; then
	NAME="${NAME:-$1}"
	OPTIONS="${OPTIONS} -p $PORT:6006"
elif [[ $1 == "/bin/bash" ]]; then
	NAME="${NAME:-bash}"
else
        NAME="${NAME:-$(TZ=UTC-9 date +%y%m%d%H%M%S)-$(openssl rand -hex 4)}"
fi

docker run -it --rm \
	-v $DATADIR:/home/$USER/data \
	-v `pwd`:$WORKDIR \
	-v /home/$USER/.cache:/home/$USER/.cache \
	--shm-size=512G ${OPTIONS} \
	--gpus $GPU \
	--name ${USER}-${PROJECT}-${NAME} $USER/$PROJECT \
	$@