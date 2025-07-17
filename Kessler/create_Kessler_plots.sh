i=0
while true; do
    echo "$i" | python3 Kessler_plot.py
    # Check the exit status. If no more bins to loop over, break the loop.
    if [ $? -ne 0 ]; then
        break
    fi
    ((i++))
done