$	??P?V?d@??}?#r@Oʤ?6 ??!?p?{?j@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?p?{?j@?0??=@1.9???X|@Il???Z3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails Oʤ?6 ???}?????1a2U0*?c?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?N?Z?7??1?h㈵?d?I?d???r11*	fffff?>@2T
Iterator::Root::ParallelMapV2+??????!5??~??O@)+??????15??~??O@:Preprocessing2E
Iterator::Root???B?i??!????tX@)??ZӼ???1K??">?@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipa2U0*?S?!H%?e@)a2U0*?S?1H%?e@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????&?#@Q'B?&?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??M??#@?g?.1@!?0??=@	!       "$	2?>???b@???P?]p@a2U0*?c?!.9???X|@*	!       2	!       :	????@???KT&@!l???Z3@B	!       J	!       R	!       Z	!       b	!       JGPUb q????&?#@y'B?&?V@