$	?w?ޜ?Q@?Q?o??c@????EB{?! ??R?u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails' ??R?u@?)???7@1?,?Qt@I?5?o??@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails eS??.????+H3??1iUMu_?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?????%~?1?????%~?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails????EB{?1????EB{?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?S?K???1??IӠhn?I?l?????r11*	??????y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap0L?
F%??!E?JԮ?K@)/n????1KԮD??H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?s??˾?!W?v%jW=@)Tt$?????1??18-@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?sF????!?????-@)?-????1v%jW?v'@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat	?c???!?]?ڕ?@)g??j+???1ծD?J?@:Preprocessing2T
Iterator::Root::ParallelMapV2lxz?,C??!?JԮD?
@)lxz?,C??1?JԮD?
@:Preprocessing2E
Iterator::Rootz6?>W??!???|@)?{??Pk??1,Q??+	@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????????!dp>?c@)????????1dp>?c@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicea??+e??!28??1@)a??+e??128??1@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??ZӼ???!?cp>?@)??ZӼ???1?cp>?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?E??????!?ڕ?]	P@);?O??n??1?????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!?v%jW???)?????g?1?v%jW???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??H?}]?!??????)??H?}]?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIp??>??@Q??<?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	bK??z?@????'?$@!?)???7@	!       "$	j־%AP@?????+b@iUMu_?!?,?Qt@*	!       2	!       :	j??ĺ'????]????!?5?o??@B	!       J	!       R	!       Z	!       b	!       JGPUb qp??>??@y??<?W@