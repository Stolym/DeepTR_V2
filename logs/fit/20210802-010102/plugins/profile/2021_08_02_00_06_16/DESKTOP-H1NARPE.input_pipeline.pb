$	-s??f?e@R??r@.2???!V???Q?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'V???Q?@?KTo?;@1?j??G?}@I?릔??-@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails .2???????ơ?1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???Ŭ?1??!??j?Iș&l???r11*	?????ov@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[Ӽ???!??|?nSK@)?ǘ?????1??4??I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?JY?8ֵ?!5?x???7@)R???Q??1@???_v*@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?ݓ??Z??!+E	??%@)z6?>W[??1??c??"@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?l??????!B?HV??$@)?{??Pk??1?~???@:Preprocessing2E
Iterator::Root7?[ A??!}?nS=?"@)Q?|a2??1H?k?f@:Preprocessing2T
Iterator::Root::ParallelMapV29??v????!c???'?@)9??v????1c???'?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ?????!*??M?@)Ǻ?????1*??M?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchvq?-??! ????@)vq?-??1 ????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???Q???!C41??P@)_?Q?{?1!^$?pN??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?q????o?!??T??a??)?q????o?1??T??a??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice-C??6j?!z:"????)-C??6j?1z:"????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!?d?????)????MbP?1?d?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIP??X[o @Q?e???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???:K?"@6'??]0@!?KTo?;@	!       "$	u/
???c@?,??jLq@rQ-"??[?!?j??G?}@*	!       2	!       :	sa????@0*?"-0!@!?릔??-@B	!       J	!       R	!       Z	!       b	!       JGPUb qP??X[o @y?e???V@