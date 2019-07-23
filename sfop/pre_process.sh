# this is a comment
# if there are comments in the second line of the image, then remove them
# because c_img library (used in SFOP) cannot read these images 
# for f in hpatches-sequences-release/*/*.ppm; do
for f in images/*.ppm; do
	second_line=$(sed -n '2p' "$f")
	echo $second_line
	if [[ $second_line == \#* ]];
	then 
		sed -i '2 d' "$f"
	fi
done
