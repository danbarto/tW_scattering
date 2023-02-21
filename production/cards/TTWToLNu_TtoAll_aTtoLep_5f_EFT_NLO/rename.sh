export OLD="TTW_5f_EFT_NLO_test"
export NEW="TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO"

for FILE in *.dat;
do
    echo $FILE;
    NEWFILE="${FILE/$OLD/$NEW}"
    echo $NEWFILE
    mv $FILE $NEWFILE
done
