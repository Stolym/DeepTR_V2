?$	?w?ޜ?Q@?Q?o??c@????EB{?! ??R?u@	!       "h
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
	bK??z?@????'?$@!?)???7@	!       "$	j־%AP@?????+b@iUMu_?!?,?Qt@*	!       2	!       :	j??ĺ'????]????!?5?o??@B	!       J	!       R	!       Z	!       b	!       JGPUb qp??>??@y??<?W@?"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_7_grad/Conv2DBackpropFilterConv2DBackpropFilter%????%??!%????%??0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_5_grad/Conv2DBackpropFilterConv2DBackpropFilterM?E???!9$?R"??0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_4_grad/Conv2DBackpropFilterConv2DBackpropFilter???????!?t?_???0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_6_grad/Conv2DBackpropFilterConv2DBackpropFilterW?6???!]v????0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_7_grad/Conv2DBackpropInputConv2DBackpropInput?W?????!EL??C???0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_4_grad/Conv2DBackpropInputConv2DBackpropInput ?F????!??????0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_5_grad/Conv2DBackpropInputConv2DBackpropInput?)a?????!?!?a[???0"?
?gradient_tape/model/conv_lstm1d_1/while/model/conv_lstm1d_1/while_grad/body/_557/gradient_tape/model/conv_lstm1d_1/while/gradients/model/conv_lstm1d_1/while/convolution_6_grad/Conv2DBackpropInputConv2DBackpropInput?\??#???!Q-??????0"g
Kmodel/conv_lstm1d_1/while/body/_279/model/conv_lstm1d_1/while/convolution_5Conv2D???ė???!}??P????"g
Kmodel/conv_lstm1d_1/while/body/_279/model/conv_lstm1d_1/while/convolution_7Conv2DIǷ+???!?g??????Q      Y@Y#??A
??a???֟X@qc??o?N@yn^??dH6?"?

both?Your program is POTENTIALLY input-bound because 6.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?60.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Maxwell)(: B 