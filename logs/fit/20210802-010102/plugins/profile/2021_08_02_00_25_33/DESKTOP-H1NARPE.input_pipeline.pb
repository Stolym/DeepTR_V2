$	?/???U@_???U?e@??g?ej??!HP???u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'HP???u@d???;@1?I?5S?s@I?<??-@@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?x????mO???~?1$D??b?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??g?ej??1??g?ej??r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?N????1
?F?s?I?<????r11*	ffffffx@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??K7?A??!?K?`EH@)??QI????1C??6??F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?E???Ը?!`mާ?8@)h??|?5??1/?u?9.@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?0?*???!z???!?$@)'?Wʢ?1?K?`?"@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??~j?t??!?!XG?w#@)??y?):??1??:?<"@:Preprocessing2T
Iterator::Root::ParallelMapV2X9??v???!i?>?%?@)X9??v???1i?>?%?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?8EGr???!XG??)?O@)?0?*???1z???!?@:Preprocessing2E
Iterator::Root46<?R??!ާ?dV&@) ?o_Ή?1??d??	@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchvq?-??!?>?%C0 @)vq?-??1?>?%C0 @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?q????o?!?!XG????)?q????o?1?!XG????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!?>?%C???)y?&1?l?1?>?%C???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!?Oq????)a2U0*?c?1?Oq????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceŏ1w-!_?!?h?>?%??)ŏ1w-!_?1?h?>?%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI0j??r)$@Q??˭?zV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@???M?+@!d???;@	!       "$	-t%??S@}>G=?c@$D??b?!?I?5S?s@*	!       2	!       :	!??^?#??S=??@!?<??-@@B	!       J	!       R	!       Z	!       b	!       JGPUb q0j??r)$@y??˭?zV@