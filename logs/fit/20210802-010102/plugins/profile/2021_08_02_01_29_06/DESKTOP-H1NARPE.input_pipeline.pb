$	???3??V@c?l?	?f@???'ׄ?!?fI??v@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?fI??v@???_v;<@1?z?bt@I4M?~2?@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?ص?ݒ???4?;???1iUMu_?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???'ׄ?1???'ׄ?r4"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!H?????1[???iv?I<g????r11*	53333?z@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapW[??????!~sSJ@)xz?,C??1?X?pl?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?(\?????!C?p?2?@)L7?A`???1'?y??/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?/?$??!`?4?P/@)?|a2U??1??s.@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?#??????!Q/i)W?@)?D???J??1??nsC@:Preprocessing2T
Iterator::Root::ParallelMapV2M??St$??!^BE?I@)M??St$??1^BE?I@:Preprocessing2E
Iterator::Root o?ŏ??!??^' @)g??j+???1F?5?
@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchŏ1w-!?!ut?%ɡ??)ŏ1w-!?1ut?%ɡ??:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??h o???!/u???\N@)y?&1?|?1?x?Q_??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicea??+ei?!)??[??)a??+ei?1)??[??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!|?XrX??)??_vOf?1|?XrX??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!??h????)??_?Le?1??h????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!?5?f????)/n??R?1?5?f????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?? u?#@Q?!?_Q?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/O??>@Y2D?o:,@!???_v;<@	!       "$	?w%cT@?D??bd@iUMu_?!?z?bt@*	!       2	!       :	'??h???#??Z@!4M?~2?@B	!       J	!       R	!       Z	!       b	!       JGPUb q?? u?#@y?!?_Q?V@