#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../


if [ "$#" -ne 4 ]; then
    echo "USAGE: run_experiment.sh TASK DATA_DIR SEED CUDA_VISIBLE_DEVICE"
    echo "TASK is the name of a task, e.g., snips."
    echo "DATA_DIR is the directory containing the data. The directory should have train.txt, dev.txt and test.txt files."
    echo "SEED is the random seed to be used."
    echo "CUDA_VISIBLE_DEVICE is the index of the cuda device where to run this experiment."
    exit
fi

TASK=$1
DATA_DIR=$2
SEED=$3

export CUDA_VISIBLE_DEVICES=$4

# Base pre-trained models
CLASSIFICATIONMODEL="bert-base-uncased"
SEQ2SEQMODEL="facebook/bart-base"

TRAINFILEPATH="${DATA_DIR}/train.txt"
DEVFILEPATH="${DATA_DIR}/dev.txt"
TESTFILEPATH="${DATA_DIR}/test.txt"

BERTEPOCH=10
PATIENCE=3
BERTGENERATED_FOLDS=5

BERT_FIRST_TRAIN_LR_LIST="5e-5"
MAXBERTGENERATEDEXAMPLES_LIST="1 2 5"

BART_LR_LIST="5e-5"
BERTGENERATEDCOSIMTHRESHOLD_LIST="0.00"

BARTEPOCHS="10"
BART_PATIENCE="3"
TOP_P="0.90"
TOP_K="100"
MAXBARTGENERATEDEXAMPLES_LIST="1 2 3 5 7 10"

OUTDIR="${TASK}_output_seed${SEED}"

for BERTGENERATEDCOSIMTHRESHOLD in ${BERTGENERATEDCOSIMTHRESHOLD_LIST}
do
  for BART_LR in ${BART_LR_LIST}
  do
    for BERT_FIRST_TRAIN_LR in ${BERT_FIRST_TRAIN_LR_LIST}
    do
      # set the LR used in the BERT training exposed to the augmented data as the
      # LR used in the first training (with the original dataset)
      BERT_SECOND_TRAIN_LR=${BERT_FIRST_TRAIN_LR}

      echo "******************************"
      echo "Start training of first classification model"
      echo "******************************"

      WORKINGDIR=${OUTDIR}/${CLASSIFICATIONMODEL}_${BERTEPOCH}epbert_${BERT_FIRST_TRAIN_LR}lr_${BERTGENERATEDCOSIMTHRESHOLD}simbert_${BART_LR}bartlr

      mkdir -p ${WORKINGDIR}

      # Tmp files where to write generated examples
      TMP_TRAIN_SIMILAR_EXAMPLE=${WORKINGDIR}/"tmp_train_expl"
      TMP_TEST_SIMILAR_EXAMPLE=${WORKINGDIR}/"tmp_test_expl"
      TRAIN_SIMILAR_EXAMPLE=${WORKINGDIR}/"train_expl"
      DECODER_INPUT=${WORKINGDIR}/"decode_input"

      # Training log file
      LOG_FIRST_TRAIN=${WORKINGDIR}/"log_original.txt"

      # Path to save the model
      OUTPUTBERTMODELFILEPATH=${WORKINGDIR}/"model_original_"${CLASSIFICATIONMODEL}".pickle"

      # Train the first bert classifiier and select the most similar examples according to CLS cosine similarity
      python -u ../dats/classifier/classifier.py -tr ${TRAINFILEPATH} \
        -dev ${DEVFILEPATH} -te ${TESTFILEPATH} -m ${OUTPUTBERTMODELFILEPATH} \
        -model ${CLASSIFICATIONMODEL} -max_gen 50 --epochs ${BERTEPOCH} --patience ${PATIENCE}\
        -tr_sim ${TMP_TRAIN_SIMILAR_EXAMPLE} -te_sim ${TMP_TEST_SIMILAR_EXAMPLE} -lr ${BERT_FIRST_TRAIN_LR} \
        -nfold ${BERTGENERATED_FOLDS} --seed ${SEED} > ${LOG_FIRST_TRAIN}

      for I_FOLD in $(seq 1 ${BERTGENERATED_FOLDS})
      do
        LIST_OF_TMP_TRAIN_FOLD=""
        LIST_OF_TMP_TEST_FOLD=""
        # REMOVE THE J_TH ELEMENT FROM THE TEST ACCORDING TO THE FOLD STRATEGY
        for J_FOLD in $(seq 1 ${BERTGENERATED_FOLDS})
        do
          if [ ${I_FOLD} -eq ${J_FOLD} ]
          then
            # ADD THE TMP FOLD TO THE TEST
            LIST_OF_TMP_TEST_FOLD=${LIST_OF_TMP_TEST_FOLD}" "${TMP_TRAIN_SIMILAR_EXAMPLE}_${J_FOLD}
          else
            # ADD THE TMP FOLD TO THE TRAIN
            LIST_OF_TMP_TRAIN_FOLD=${LIST_OF_TMP_TRAIN_FOLD}" "${TMP_TRAIN_SIMILAR_EXAMPLE}_${J_FOLD}
          fi
        done
        # WRITE TRAIN FILE FOR THE SEQ2SEQ MODEL
        cat ${LIST_OF_TMP_TRAIN_FOLD} > ${TRAIN_SIMILAR_EXAMPLE}"_"${I_FOLD}"fold"
        # WRITE INFPUT FILE FOR THE DECODE step
        cat ${LIST_OF_TMP_TEST_FOLD} | awk -F"\t" '{print $3,$4}' | uniq > ${DECODER_INPUT}"_"${I_FOLD}"fold"
      done

      ####################
      # SEQ2SEQ Model Training
      ####################
      echo "******************************"
      echo "Start training of the seq2seq model"
      echo "******************************"
      for MAXBERTGENERATEDEXAMPLES in ${MAXBERTGENERATEDEXAMPLES_LIST}
      do
        # The original python generated up to 50 similar example for each instance.
        # Here a subset from the variable MAXBERTGENERATEDEXAMPLES is selected
        for I_FOLD in $(seq 1 ${BERTGENERATED_FOLDS})
        do
          KTRAIN_SIMILAR_EXAMPLE=${TRAIN_SIMILAR_EXAMPLE}"_"${I_FOLD}"fold_"${MAXBERTGENERATEDEXAMPLES}"simbert.txt"
          cat ${TRAIN_SIMILAR_EXAMPLE}"_"${I_FOLD}"fold" | awk -F"\t" -v k=${MAXBERTGENERATEDEXAMPLES} '{if($2<=k) print }' > ${KTRAIN_SIMILAR_EXAMPLE}
        done

        # Train the seq2seq model
        for BARTEPOCH in ${BARTEPOCHS}
        do
          # Train on each of the k fold
          for I_FOLD in $(seq 1 ${BERTGENERATED_FOLDS})
          do
            # TRAIN BART
            KTRAIN_SIMILAR_EXAMPLE=${TRAIN_SIMILAR_EXAMPLE}"_"${I_FOLD}"fold_"${MAXBERTGENERATEDEXAMPLES}"simbert.txt"
            OUTPUTBARTMODELFILEPATH=${WORKINGDIR}/"bart_model_"${I_FOLD}"fold_"${MAXBERTGENERATEDEXAMPLES}"simbert_"${BARTEPOCH}"epbart"

            python -u ../dats/seq2seq/seq2seq_train.py -i ${KTRAIN_SIMILAR_EXAMPLE} -m ${OUTPUTBARTMODELFILEPATH} -epoch ${BARTEPOCH} -model_name ${SEQ2SEQMODEL} -th ${BERTGENERATEDCOSIMTHRESHOLD} -lr ${BART_LR} -p ${BART_PATIENCE} --seed ${SEED}
          done

          # For each fold generate a number of augmented examples as specified in the variable MAXBARTGENERATEDEXAMPLES
          for MAXBARTGENERATEDEXAMPLES in ${MAXBARTGENERATEDEXAMPLES_LIST}
          do
            AUGMENTEDTRAINFILEPATH=${WORKINGDIR}/"augmented_train_"${MAXBERTGENERATEDEXAMPLES}"simbert_"${BARTEPOCH}"epbart_"${MAXBARTGENERATEDEXAMPLES}"simbart.txt"
            TMP_FILE_LIST=""

            for I_FOLD in $(seq 1 ${BERTGENERATED_FOLDS})
            do
              NEWTRAINFILEPATH=${WORKINGDIR}/"new_train_"${I_FOLD}"fold_"${MAXBERTGENERATEDEXAMPLES}"simbert_"${BARTEPOCH}"epbart_"${MAXBARTGENERATEDEXAMPLES}"simbart.txt"
              OUTPUTBARTMODELFILEPATH=${WORKINGDIR}/"bart_model_"${I_FOLD}"fold_"${MAXBERTGENERATEDEXAMPLES}"simbert_"${BARTEPOCH}"epbart"
              TMP_FILE_LIST=${TMP_FILE_LIST}" "${NEWTRAINFILEPATH}

              python -u ../dats/seq2seq/seq2seq_predict.py -i ${DECODER_INPUT}"_"${I_FOLD}"fold" -n ${MAXBARTGENERATEDEXAMPLES} -m ${OUTPUTBARTMODELFILEPATH}/best_model -o ${NEWTRAINFILEPATH} --do_sample True --top_p ${TOP_P} --top_k ${TOP_K} --seed ${SEED}
            done

            echo ${TMP_FILE_LIST}
            cat ${TMP_FILE_LIST} | awk -F"\t" '{print $1,$3}' | sort | uniq > ${AUGMENTEDTRAINFILEPATH}

            OUTPUTAUGMBERTMODELFILEPATH=${WORKINGDIR}/"model_augmented_"${CLASSIFICATIONMODEL}"_"${MAXBERTGENERATEDEXAMPLES}"simbert_"${BARTEPOCH}"epbart_"${MAXBARTGENERATEDEXAMPLES}"simbart_"${PRETRAIN_EPOCH}"pt.pickle"
            LOG_SECOND_TRAIN=${WORKINGDIR}/"log_augmented_"${CLASSIFICATIONMODEL}"_"${MAXBERTGENERATEDEXAMPLES}"simbert_"${BARTEPOCH}"epbart_"${MAXBARTGENERATEDEXAMPLES}"simbart.txt"

            # Train final version of bert model using also the generated training material
            python -u ../dats/classifier/classifier.py -tr ${AUGMENTEDTRAINFILEPATH} ${TRAINFILEPATH} -dev ${DEVFILEPATH} -te ${TESTFILEPATH} -m ${OUTPUTAUGMBERTMODELFILEPATH} -model ${CLASSIFICATIONMODEL} -max_gen 0 --epochs ${BERTEPOCH} --patience ${PATIENCE} -lr ${BERT_SECOND_TRAIN_LR} --seed ${SEED} > ${LOG_SECOND_TRAIN}

            rm ${WORKINGDIR}/*.pickle
          done
        done

        # Removing models to save disk space
#        rm ${WORKINGDIR}/bart*/pytorch_model.bin
#        rm ${WORKINGDIR}/bart*/best_model/*

      done
    done
  done
done
