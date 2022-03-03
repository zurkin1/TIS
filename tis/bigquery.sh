#!/bin/bash


FILES="output_immunoSEQ_786_50new.tsv
output_immunoSEQ_172-197new.tsv
output_immunoSEQ_3-24new.tsv
output_immunoSEQ_26-48new.tsv
output_immunoSEQ_49new.tsv
output_immunoSEQ_786_100new.tsv
output_immunoSEQ_786_150new.tsv
output_immunoSEQ_786_200new.tsv
output_immunoSEQ_786_250new.tsv
output_immunoSEQ_786_300new.tsv
output_immunoSEQ_786_350new.tsv
output_immunoSEQ_786_400new.tsv
output_immunoSEQ_786_450new.tsv
output_immunoSEQ_786_500new.tsv
output_immunoSEQ_786_550new.tsv
output_immunoSEQ_786_600new.tsv
output_immunoSEQ_786_650new.tsv
output_immunoSEQ_786_700new.tsv
output_immunoSEQ_786_786new.tsv
output_immunoSEQ_88new.tsv
output_immunoSEQ_92-115new.tsv
output_ncbi-1new-new.tsv
output_ncbi-2new-new.tsv
output_ncbi-3new-new.tsv
output_ncbi-4new-new.tsv
output_ncbi-5new-new.tsv
output_ncbi-6new-new.tsv
output_ncbi-7new-new.tsv
output_ncbi-8new-new.tsv
output_ncbi-9new-new.tsv"


for f in $FILES
do
	echo $f
	scp dani@zelda.ls.biu.ac.il:../sarit/final_output/$f .
	bq load --source_format=CSV -F="tab" --skip_leading_rows=1 --max_bad_records=10 --allow_jagged_rows=true cdr3_dataset.cdr3 $f
	rm $f
done
