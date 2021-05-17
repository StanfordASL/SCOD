set -e 
target=''

print_usage() {
  printf "Must specify  target directory:"
  printf "   ./download_datasets.sh -t ~/datasets"
}

while getopts 't:' flag; do
  case "${flag}" in
    t) target="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

mkdir -p "$target"

cd $target
target=$(pwd)
printf "Target directory: $target"
printf "\n"

# Wine
if [ ! -d wine ]; then
    mkdir wine
    cd wine
    curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
    cd target
else
    printf "${target}/wine already exists, skipping. Remove the directory if you wish to redownload the dataset.\n"
fi

# TinyImageNet
if [ ! -d tiny-imagenet-200 ]; then
    curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
    cd tiny-imagenet-200/train
    for DIR in $(ls); do
       cd $DIR
       rm *.txt
       mv images/* .
       rm -r images
       for f in *.JPEG; do 
            mv -- "$f" "${f%.JPEG}.jpg"
        done
       cd ..
    done
    
    cd ../val
    annotate_file="val_annotations.txt"
    length=$(cat $annotate_file | wc -l)
    for i in $(seq 1 $length); do
        # fetch i th line
        line=$(sed -n ${i}p $annotate_file)
        # get file name and directory name
        file=$(echo $line | cut -f1 -d" " )
        directory=$(echo $line | cut -f2 -d" ")
        mkdir -p $directory
        mv -- "images/$file" "$directory/""${file%.JPEG}.jpg"
    done
    rm -r images
    cd $target
else
    printf "${target}/tiny-imagenet-200 already exists, skipping. Remove the directory if you wish to redownload the dataset.\n"
fi
