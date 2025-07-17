# this code will select conformal prediction plots at random
input_file="congestus_test" # name of the dataset to test (without .nc) SET BY USER
n="$(ncdump -h $input_file.nc \
  | awk '/loc =/ { gsub(/[^0-9]/,"",$3); print $3 }')" # number of elements in dataset
percent=10     # percent to select, SET BY USER
PATH_NAME="$input_file/cpSINDy/cv+"

# integer arithmetic
k=$(( n * percent / 100 ))
# guard: request at least one sample if the percent rounds to 0
(( k < 1 )) && k=1

# shuffle, sample, and open
shuf -i 0-$((n-1)) -n "$k" | while read -r idx; do
  pdf="$PATH_NAME/intervals_${idx}.pdf"

  if [[ -f "$pdf" ]]; then
    echo "Opening $pdf"
    code "$pdf" &
  else
    echo "WARNING: file not found: $pdf" >&2
  fi
done