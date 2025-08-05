cd ~/Kessler

python3 cp_test.py box64 -m full > data/box64/cpSINDy/full_test.txt
python3 cp_test.py box64 -m split > data/box64/cpSINDy/split_test.txt
python3 cp_test.py box64 -m jackknife > data/box64/cpSINDy/jackknife_test.txt
python3 cp_test.py box64 -m cv+ > data/box64/cpSINDy/cv+_test.txt

python3 cp_test.py filtered_combined_100m_full -m full > data/filtered_combined_100m_full/cpSINDy/full_test.txt
python3 cp_test.py filtered_combined_100m_full -m split > data/filtered_combined_100m_full/cpSINDy/split_test.txt
python3 cp_test.py filtered_combined_100m_full -m jackknife > data/filtered_combined_100m_full/cpSINDy/jackknife_test.txt
python3 cp_test.py filtered_combined_100m_full -m cv+ > data/filtered_combined_100m_full/cpSINDy/cv+_test.txt

python3 cp_test.py congestus_test -m full > data/congestus_test/cpSINDy/full_test.txt
python3 cp_test.py congestus_test -m split > data/congestus_test/cpSINDy/split_test.txt
python3 cp_test.py congestus_test -m jackknife > data/congestus_test/cpSINDy/jackknife_test.txt
python3 cp_test.py congestus_test -m cv+ > data/congestus_test/cpSINDy/cv+_test.txt

python3 cp_test.py rico_test -m full > data/rico_test/cpSINDy/full_test.txt
python3 cp_test.py rico_test -m split > data/rico_test/cpSINDy/split_test.txt
python3 cp_test.py rico_test -m jackknife > data/rico_test/cpSINDy/jackknife_test.txt
python3 cp_test.py rico_test -m cv+ > data/rico_test/cpSINDy/cv+_test.txt

python3 congestus_on_rico_cp_test.py -m full > data/congestus_on_rico/cpSINDy/full_test.txt
python3 congestus_on_rico_cp_test.py -m split > data/congestus_on_rico/cpSINDy/split_test.txt
python3 congestus_on_rico_cp_test.py -m jackknife > data/congestus_on_rico/cpSINDy/jackknife_test.txt
python3 congestus_on_rico_cp_test.py -m cv+ > data/congestus_on_rico/cpSINDy/cv+_test.txt

python3 congestus_on_rico_split_cp_test.py > data/congestus_on_rico_split/test.txt