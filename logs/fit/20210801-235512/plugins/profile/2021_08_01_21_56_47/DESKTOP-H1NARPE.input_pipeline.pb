$	??ǿ??d@?up%nr@?y?Cn?{?!???qd@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???qd@5??6??;@1??????|@I?u???-@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?y?Cn?{???Os?"s?1??????`?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????뉞?Ҭl????1??9̗g?r11*	33333?6@2T
Iterator::Root::ParallelMapV246<?R??!???hAH@)46<?R??1???hAH@:Preprocessing2E
Iterator::Rootj?t???!?f׋??W@)?g??s???1H@ͮYG@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipa2U0*?S?!???B7%@)a2U0*?S?1???B7%@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?
~6? @Q?>0>?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???>?"@x9?۰0@??Os?"s?!5??6??;@	!       "$	?/?^'c@???_?p@??????`?!??????|@*	!       2	!       :	????D?@Ⱥ??f
!@!?u???-@B	!       J	!       R	!       Z	!       b	!       JGPUb q?
~6? @y?>0>?V@