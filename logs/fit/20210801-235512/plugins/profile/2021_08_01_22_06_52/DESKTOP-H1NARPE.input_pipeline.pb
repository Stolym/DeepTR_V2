$	6??6?N??A?'9>??????W?x?!l???D??	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails l???D???-???1?mO???^?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???W?x?1???W?x?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
??O?my?1
??O?my?r5"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsR???Tz?1R???Tz?r6*	fffff?D@2T
Iterator::Root::ParallelMapV2?g??s???!?л?u?I@)?g??s???1?л?u?I@:Preprocessing2E
Iterator::Root#??~j???!??1p?X@)?N@aÓ?1\\??]G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!X??#???)-C??6J?1X??#???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 74.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIv|?]??R@Q)f???9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?-????-???!?-???	!       "$	???|	u?x-???a??mO???^?!R???Tz?*	!       2	!       :	!       B	!       J	!       R	!       Z	!       b	!       JGPUb qv|?]??R@y)f???9@