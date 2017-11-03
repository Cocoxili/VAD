for I in {0..24};do
    # echo $I
    copy-feats scp:feats_$I.scp ark:- | apply-cmvn  --norm-means=true --norm-vars=false  scp:cmvn.scp ark:- ark:- | add-deltas --delta-order=1 ark:- ark,t:raw_feat.$I.txt 
done
