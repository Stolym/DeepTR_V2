$	ymI!??c@v??q-5q@?d??7i??!?ߠ?z?}@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?ߠ?z?}@p$?`S?9@1?(]??t{@I??֦??'@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?d??7i???[?????1$D??b?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?4?;???1?'eRC[?I??? !ʧ?r11*	23333?D@2T
Iterator::Root::ParallelMapV2?z6?>??!?mD0?K@)?z6?>??1?mD0?K@:Preprocessing2E
Iterator::Root?/?'??!K?V?ͅX@)HP?sג?1?i? ?E@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!^mOj????)-C??6J?1^mOj????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?r?Yv?@Q?`?xW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f?7?>!@?L?}??-@!p$?`S?9@	!       "$	?)?9?Mb@6A7???o@?'eRC[?!?(]??t{@*	!       2	!       :	?N?
?|@S??!?@!??֦??'@B	!       J	!       R	!       Z	!       b	!       JGPUb q?r?Yv?@y?`?xW@