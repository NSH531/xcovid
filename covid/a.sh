for d in ~/covid/nii/*.nii.gz ; do
    python3 ~/covid/nii.py -i $d -o ~/covid/images_png
    echo "$d"
done