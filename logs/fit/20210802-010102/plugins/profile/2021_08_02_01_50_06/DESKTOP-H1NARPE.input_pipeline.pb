$	????5T@:Ɠ[?4d@?30??&v?!?????4t@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?????4t@? {?<@1FA??Qr@I~?N?Z/??r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?lt?Oq??D???XP??1?3?ۃ`?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?30??&v?1?30??&v?r6"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?}?֤ۢ?Tb.???1?mO???^?r11*	    H@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?!??u???!"???WJ@)]?Fx??1D?,?I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???<,???!?-??c=@)????(??1?}M???6@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat]m???{??!X????@)aTR'????1??D{n?@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?St$????!οBC??@)?U???؟?1?0????@:Preprocessing2T
Iterator::Root::ParallelMapV2?+e?X??!??!u?8@)?+e?X??1??!u?8@:Preprocessing2E
Iterator::Root'???????!???vW!@){?G?z??1	.l???@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZippΈ?????!I?5?jO@)??_vO??1??οBC@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?ZӼ?}?!?S?m????)?ZӼ?}?1?S?m????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice"??u??q?!?Z??~??)"??u??q?1?Z??~??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!o?!0V?F?K??)ŏ1w-!o?10V?F?K??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?J?4a?!x??g???)?J?4a?1x??g???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!C??q?u??)-C??6Z?1C??q?u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI_????"@QT? ʩV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??$>w?@?? ]}?,@!? {?<@	!       "$	y=?RR@|~9?Qb@?mO???^?!FA??Qr@*	!       2	!       :	~?N?Z/??~?N?Z/??!~?N?Z/??B	!       J	!       R	!       Z	!       b	!       JGPUb q_????"@yT? ʩV@