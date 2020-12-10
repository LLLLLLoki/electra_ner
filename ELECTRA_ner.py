import tensorflow as tf
import modeling
import optimization
import tokenization
from random import shuffle
import collections
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_dict(path):
    char2num = dict()
    num2char = dict()

    with open(path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        for i ,line in enumerate(lines):
            char = line.strip()
            char2num[char] = i
            num2char[i] = char
        return char2num,num2char

def get_config(path):
    with open(path,'r',encoding= 'utf-8') as f:
        config = json.load(f)

    label2num,num2label = get_dict(config['label_dir'])
    # vocab2num,num2vocab = get_dict(config['vocab'])
    return config,label2num,num2label

class InputExample(object):
    def __init__(self,guid,text,lens,label = None):
        self.guid = guid
        self.text = text
        self.lens = lens
        self.label = label

class InputFeatures(object):
    def __init__(self,input_ids,input_mask,lens,label_ids = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.lens = lens
        self.label_ids = label_ids

class DataProcessor(object):
    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_test_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_examples(self,data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls,input_file):
        data,chars,labels = [],[],[]

        with tf.gfile.Open(input_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    char,label = line.rstrip('\n').split('\t')
                    chars.append(char)
                    labels.append(label)

                else:
                    lens = len(chars)
                    data.append([chars,labels,lens])
                    chars,labels = [],[]
            return data

class ELECTRA_NerProcessor(DataProcessor):
    def get_train_examples(self,data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir,'train.txt')),'train'
        )

    def get_test_examples(self,data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir,'test.txt')),'test'
        )

    def get_dev_examples(self,data_dir):
        return self._create_examples(
            self._read_txt(os.path.join(data_dir,'dev.txt'))
        )

    def _create_examples(self,datas,set_type):
        examples = []
        print('length of lines:',len(datas))
        if set_type is 'train':
            shuffle(datas)

        for (i,line) in enumerate(datas):
            guid = "%s-%s"%(set_type,i)
            text = line[0]
            label = line[1]
            lens = line[2]

            examples.append(
                InputExample(
                    guid = guid,text = text,lens = lens,label = label
                )
            )
        return examples

def convert_single_example(config,ex_index,example,tokenizer,label2num):
    tokens_ = example.text
    labels_ = example.label
    lens = example.lens
    if len(tokens_) > config['max_length'] - 2:
        tokens_ = tokens_[0:(config['max_length']-2)]
        labels_ = labels_[0:(config['max_length']-2)]
        lens = config['max_length']
    else:
        lens = lens + 2

    tokens = []
    labels = []
    label_ids = []
    tokens.append('[CLS]')
    labels.append('O')
    for token in tokens_:
        tokens.append(token)

    for label in labels_:
        labels.append(label)

    tokens.append('[SEP]')
    labels.append('O')
    for label in labels:
        label_ids.append(label2num[label])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < config['max_length']:
        input_ids.append(0)
        input_mask.append(0)
    assert len(input_ids) == config['max_length']
    assert len(input_mask) == config['max_length']

    if ex_index < 5:
        tf.logging.info('***Example***')
        tf.loggint.info('guid:%s'%(example.guid))
        tf.logging.info('tokens:%s'%''.join([x for x in tokens]))
        tf.logging.info('input_ids'%' '.join(str(x) for x in input_ids))
        tf.logging.info('input_mask:%s'%' '.join(str(x) for x in input_mask))
        tf.logging.info('label:%s(id = %s)'%(labels,label_ids))

    feature = InputFeatures(
        input_ids = input_ids,
        lens = lens,
        label_ids = label_ids,
        innput_mask = input_mask
    )
    return feature

def file_based_convert_examples_to_features(examples,config,label2num,output_file,tokenizer = None):
    writer = tf.python_io.TFRecordWriter(output_file)
    lendata = len(examples)
    for (ex_index,example) in enumerate(examples):
        if ((ex_index * 100)/lendata) % 10 == 0:
            tf.logging.info('Writing example %d of %d'%(ex_index,lendata))

        feature = convert_single_example(config,ex_index,example,tokenizer)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list = tf.train.Int64List(value = list(values)))
            return f

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_int_feature(feature.input_mask)
        features['lens'] = create_int_feature(feature.lens)
        features['label_ids'] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features = tf.train.Features(feature = features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_base_input_fn_builder(config,input_file,is_training,drop_remainder):
    name_to_features = {
        'input_ids':tf.FixedLenFeature([config['max_length']],tf.int64),
        'input_mask':tf.FixedLenFeature([config['max_length']],tf.int64),
        'lens':tf.FixedLenFeature([],tf.int64),
        'label_ids':tf.FixedLenFeature([config['max_length']],tf.int64)
    }

    def _decode_record(record,name_to_features):
        example = tf.parse_single_example(record,name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
                example[name] = t
        return example

    def input_fn(params):
        batch_size = params['batch_size']
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size = 1024)

        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record:_decode_record(record,name_to_features),
            batch_size = batch_size,
            drop_remainder = drop_remainder
        ))
        return d
    return input_fn

def create_model(config,bert_config ,is_training,input_ids,
                 input_mask,lens,label_ids = None):

    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = input_mask,
        use_one_hot_embeddings=config['use_one_hot_embeddings']
    )
    output_layer = model.get_sequence_output()
    with tf.variable_scope('logits'):
        if is_training:
            output_layer = tf.nn.dropout(output_layer,keep_prob=config['dropout'])
        logits = tf.layers.dense(output_layer,config['label_num'])

    with tf.variable_scope('crf'):
        transition_params = tf.get_variable(
            name = 'trans',shape = [config['label_num'],config['label_num']],
            initializer=modeling.create_initializer(config['initializer_range'])
        )

        loss ,_= tf.contrib.crf.crf_log_likelihood(
            logits,label_ids,lens,transition_params
        )

        decode_tags ,_ = tf.contrib.crf.crf_decode(
            logits,transition_params,lens
        )
        return decode_tags,loss

def model_fn_builder(config,bert_config,num_train_steps,num_warmup_steps):
    def model_fn(features,labels,mode,params):
        tf.logging.info('***Features***')
        for name in sorted(features.keys()):
            tf.logging.info('name = %s,shape = %s'%(name ,features[name].shape))

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        lens = features['lens']
        label_ids = features['label_ids']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        pred_ids,loss = create_model(
            config,bert_config,is_training,
            input_ids,input_mask,lens,label_ids
        )
        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if config['init_checkpoint']:
            (assignment_map,initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars,config['init_checkpoint']
            )
            tf.train.init_from_checkpoint(config['inti_checkpoint'],assignment_map)
        tf.logging.info('***Trainavble Variables***')

        for var in tvars:
            init_string = ''
            if var.name in assignment_map:
                init_string = ',* INIT_FROM_CKPT*'
            tf.logging.info('name = %s,shape = %s%s',var.name,var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            electra_tvars = [x for x in tvars if 'electra' in x.name]
            finetuning_tvars = [x for x in tvars if 'electra' not in x.name]

            electra_op = optimization.create_optimizer(
                loss = loss,learning_rate = config['electra_lreatning_rate'],
                num_train_steps = num_train_steps,warmup_steps = num_warmup_steps,
                variable_list = electra_tvars
            )

            finetuning_op = optimization.create_optimizer(
                loss=loss, learning_rate=config['finetuning_lreatning_rate'],
                num_train_steps=num_train_steps, warmup_steps=num_warmup_steps,
                variable_list=finetuning_tvars
            )

            train_op = tf.group(electra_op,finetuning_op)

            hook_dict = {}
            hook_dict['loss'] = loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict,every_n_iter=config['save_summary_steps']
            )

            output_spec = tf.estimator.EstimatorSpec(
                mode = mode,
                loss = loss,
                train_op = train_op,
                training_hooks = [logging_hook]
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids,pred_ids,label_num,input_mask):
                cm = metrics.streaming_confusion_matrix(label_ids,pred_ids,
                                                        label_num-1,input_mask)
                return {
                    'confusion_matrix':cm
                }
            eval_metrics = (metric_fn,[label_ids,pred_ids,config['label_num'],input_mask])
            output_spec = tf.estimator.EstimatorSpec(
                mode = mode,
                loss = loss,
                eval_metric_ops=eval_metrics
            )

        else:
            predictions = {'pred_ids':pred_ids}
            output = {'serving_default':tf.estimator.export.PredictOutput(predictions)}
            output_spec = tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = predictions,
                export_outputs = output
            )
        return output_spec
    return model_fn

def main(path):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = ELECTRA_NerProcessor()

    config,label2num,num2label = get_config(path)
    bert_config = modeling.BertConfig.from_json_file(config['bert_config_file'])
    #bert_config = modeling.BertConfig.from_json_file(config['bert_config_file'])
    tokenizer = tokenization.FullTokenizer(
        vocab_file=config['bert_vocab'],do_lower_case=config['do_lower_case']
    )

    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])

    session_config = tf.ConfigProto(
        log_device_placement = False,
        inter_op_parallelism_threads = 0,
        intra_op_parallelism_threads = 0,
        allow_soft_palcement = True
    )

    run_config = tf.estimator.RunConfig(
        model_dir = config['output'],
        save_summary_steps = config['save_summary_steps'],
        save_checkpoints_steps = config['save_ckpt_steps'],
        session_config = session_config
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if config['do_train']:
        train_examples = processors.get_train_examples(config['data_dir'])
        num_train_steps = int(
            len(train_examples) * 1.0 /config['batch_size']
        )
        if num_train_steps <1:
            raise AttributeError('trainin data is so small......')

        num_warmup_steps = int(num_train_steps * config['warmup_proportion'])
        tf.logging.info('***Running training***')
        tf.logging.info('Num examples = %d',len(train_examples))
        tf.logging.info('batch size = %d',config['batch_size'])
        tf.logging.info('Num steps = %d',num_train_steps)

    if config['do_eval']:
        eval_examples = processors.get_dev_examples((config['data_dir']))
        tf.logging.info('***Running evaluation***')
        tf.logging.info('Num examples = %d',len(eval_examples))
        tf.logging.info('Batch size = %d',config['batch_size'])

    model_fn = model_fn_builder(
        config,bert_config,num_train_steps,num_warmup_steps
    )
    params = {
        'batch_size':config['batcvh_size']
    }
    estimator = tf.estimator.Estimator(
        model_fn,
        params = params,
        config = run_config
    )

    if config['do_train'] and config['do_eval']:
        train_file = os.path.join(config['output'],'train.tf_record')
        if not os.path.exists(train_file):
            file_based_convert_examples_to_features(
                train_examples,config,label2num,
                train_file,tokenizer
            )

        train_input_fn = file_base_input_fn_builder(
            config = config,
            input_file = train_file,
            is_training=True,
            drop_remainder=True
        )
        eval_file = os.path.join(config['output_dir'],'eval.tf_record')
        if not os.path.exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, config, label2num,
                eval_file, tokenizer
            )
        eval_input_fn = file_base_input_fn_builder(
            config = config,
            input_file = eval_file,
            is_training = False,
            drop_remainder=False
        )

        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator = estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps = 0,
            run_every_secs = None,
            run_every_steps = config['save_ckpt_steps']
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn = train_input_fn,
            max_steps = num_train_steps,
            hooks = [early_stopping_hook]
        )

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)

    if config['do_predict']:
        predict_examples = processors.get_test_examples(config['data_dir'])
        predict_file = os.path.join(config['output_dir'],'predict.tf_record')
        file_based_convert_examples_to_features(
            predict_examples,
            config,
            label2num,
            predict_file,
            tokenizer
        )

        tf.logging.info('***Running Prediction***')
        tf.logging.info('Num examples = %d',len(predict_examples))
        tf.logging.info('Batch size = %d',config['batch_size'])
        predict_input_fn = file_base_input_fn_builder(
            config = config,
            input_file=predict_file,
            is_training=False,
            drop_remainder=False
        )

        result = estimator.predict(input_fn = predict_input_fn)
        output_predict_file=os.path.join(config['output_dir'],'label_test.txt')
        def result_to_pair(writer):
            for predict_line ,prediction in zip(predict_examples,result):
                idx=0
                line=''
                line_token=predict_line.text_a
                label_token=predict_line.label
                len_seq = predict_line.lens-1

                if len(line_token)!=len(label_token):
                    tf.logging.info(predict_line.text_a)
                    tf.logging.info(predict_line.label)
                    break
                for i , id in enumerate(prediction['pred_ids'][1:len_seq]):
                    curr_labels=num2label[id]
                    try:
                        line += line_token[i] + '\t' + label_token[i] + '\t' +curr_labels +'\n'
                    except Exception as e:
                        tf.logging.info(e)
                        tf.logging.info(predict_line.text_a)
                        tf.logging.info(predict_line.label)
                        line = ''
                        break
                    idx +=1
                writer.write(line + '\n')

        with open(output_predict_file,'w',encoding='utf-8') as writer:
            result_to_pair(writer)
        import conlleval
        eval_result=conlleval.return_report(output_predict_file)
        print(''.join(eval_result))
        with open(os.path.join(config['output_dir'],'predict_score.txt'),'a',encoding='utf-8') as fd:
            fd.write(''.join(eval_result))

if __name__=='__main__':
    path = './data/train_config.json'
    main(path)































