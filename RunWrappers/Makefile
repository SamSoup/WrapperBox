toxigen:
	python3 main.py \
		--datasets toxigen \
		--models bart_large deberta_large flan_t5_large \
		--wrappers original KNN LGBM LMeans \
		--seed 42 \
		--pooler mean_with_attention \
		--layers 25 24 49 \
		--metrics accuracy precision recall f1 \
		--output_dir ./results

toxigen-dt:
	python3 main.py \
		--datasets toxigen \
		--models bart_large deberta_large flan_t5_large \
		--wrappers original KNN DecisionTree LGBM LMeans \
		--seed 42 \
		--pooler mean_with_attention \
		--layers 25 24 49 \
		--metrics accuracy precision recall f1 \
		--output_dir ./results

esnli:
	python3 main.py \
		--datasets esnli \
		--models bart_large deberta_large flan_t5_large \
		--wrappers original KNN LGBM LMeans \
		--seed 42 \
		--pooler mean_with_attention \
		--layers 25 24 49 \
		--metrics accuracy macro_precision macro_recall macro_f1 \
		--output_dir ./results

