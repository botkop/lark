#!/bin/bash

sr=32000
target=/home/koen/projects/botkop/lark/data/birdclef-2021/train_short_audio.wav

cd data/birdclef-2021/train_short_audio
for d in *
do 
	echo "processing $d"
	mkdir -p $target/$d
	cd $d 
	for ogg in *.ogg
	do 
		bn=$(basename $ogg .ogg)
		ffmpeg -y -hide_banner -loglevel panic -i $ogg -ac 1 -ar $sr $target/$d/$bn.wav
	done
	cd ..
done

