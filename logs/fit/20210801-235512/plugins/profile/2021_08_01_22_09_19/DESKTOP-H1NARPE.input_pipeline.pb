$	\@h=\p@	??'\?v@??z?ю??!f??%?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'f??%?@?L?n?8@1??#?}@I??_??-@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??z?ю???????ə?1? 3??O\?r11*	     ?A@2T
Iterator::Root::ParallelMapV2??d?`T??!%I?$I?I@)??d?`T??1%I?$I?I@:Preprocessing2E
Iterator::Root?? ?rh??!?$I?$IX@)??ܵ?|??1      G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????MbP?!?m۶m?@)????MbP?1?m۶m?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??
2,?@QT?<W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??%???(@:??T??1@?????ə?!?L?n?8@	!       "$	?(?7?m@J3T??u@? 3??O\?!??#?}@*	!       2	!       :	??_??@s?%c??$@!??_??-@B	!       J	!       R	!       Z	!       b	!       JGPUb q??
2,?@yT?<W@