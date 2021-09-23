for out in "/data/thar011/out/unifiedqa_bart_large_v3" "/data/thar011/out/unifiedqa_2gputest_from_uqackpt"
do
    echo "Running testwithparam.sh with param $out ..."
    bash testwithparam.sh $out
done


