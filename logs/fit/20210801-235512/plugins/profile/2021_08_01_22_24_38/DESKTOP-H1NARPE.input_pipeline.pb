$	A?} ҇f@???M?s@yxρ???!4?k??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'4?k??@f/?>@14??D@Ix(
??\*@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails yxρ?????q?@H??1???%f?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!^c?????1rQ-"??k?I??????r11*	???????@2T
Iterator::Root::ParallelMapV2K?=?U??!kq?w5H@)K?=?U??1kq?w5H@:Preprocessing2E
Iterator::Rootŏ1w-!??!?JG?X@)?W[?????1ɞ??t?G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipa2U0*?S?!???7a@)a2U0*?S?1???7a@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIh??F @Q???>?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??i*?$@C3???_1@!f/?>@	!       "$	?6LG??d@Q??­?q@???%f?!4??D@*	!       2	!       :	?h`њ?@<?`\Bh@!x(
??\*@B	!       J	!       R	!       Z	!       b	!       JGPUb qh??F @y???>?V@