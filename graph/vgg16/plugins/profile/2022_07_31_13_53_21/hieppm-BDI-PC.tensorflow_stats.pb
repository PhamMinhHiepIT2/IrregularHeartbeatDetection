"�K
 DeviceIDLE"IDLE�Unknown
BHostIDLE"IDLE1^�I�rZAA^�I�rZAa�u� +S�?i�u� +S�?�Unknown
kHost_FusedConv2D"block4_conv3/Relu(1��/!�A9��/!�AA��/!�AI��/!�Aa�7��I�?i�.��zM�?�Unknown
kHost_FusedConv2D"block4_conv2/Relu(1���x�[A9���x�[AA���x�[AI���x�[Aa��:���?i�;d��A�?�Unknown
kHost_FusedConv2D"block1_conv2/Relu(1#��~��
kHost_FusedConv2D"block2_conv2/Relu(1�O���QA9�O���QAA�O���QAI�O���QAa]��F,s�?iK�,���?�Unknown
kHost_FusedConv2D"block3_conv2/Relu(1-�����
A9-�����
AA-�����
AI-�����
Aa-��D�?il��S���?�Unknown
kHost_FusedConv2D"block3_conv3/Relu(1��~j��
A9��~j��
AA��~j��
AI��~j��
Aa��i�s�?i(��{��?�Unknown
k	Host_FusedConv2D"block4_conv1/Relu(1�S㥱��@9�S㥱��@A�S㥱��@I�S㥱��@a�.ub���?i��m{B��?�Unknown
k
Host_FusedConv2D"block2_conv1/Relu(1�O����@9�O����@A�O����@I�O����@a��4H �?i�!A��Q�?�Unknown
kHost_FusedConv2D"block3_conv1/Relu(1y�&1��@9y�&1��@Ay�&1��@Iy�&1��@a�����B�?id��/ϲ�?�Unknown
kHost_FusedConv2D"block5_conv2/Relu(1V-��N�@9V-��N�@AV-��N�@IV-��N�@ah�چ��?i�*J���?�Unknown
k
kHost_FusedConv2D"block5_conv3/Relu(1�ʡEʊ�@9�ʡEʊ�@A�ʡEʊ�@I�ʡEʊ�@a����u}?iK����t�?�Unknown
hHostMaxPool"block1_pool/MaxPool(1�"��J��@9�"��J��@A�"��J��@I�"��J��@a˧6��z?i��9���?�Unknown
kHost_FusedConv2D"block1_conv1/Relu(1`��"���@9`��"���@A`��"���@I`��"���@a��X��at?i ��p���?�Unknown
hHostMaxPool"block2_pool/MaxPool(1X9��ff�@9X9��ff�@AX9��ff�@IX9��ff�@aԻ^�5�i?i�a�=��?�Unknown
hHostMaxPool"block3_pool/MaxPool(1����l/�@9����l/�@A����l/�@I����l/�@a���tX?iUJ�x��?�Unknown
hHostMaxPool"block4_pool/MaxPool(1ףp=j��@9ףp=j��@Aףp=j��@Iףp=j��@a9HMS��G?i���m��?�Unknown
hHostMaxPool"block5_pool/MaxPool(1#��~��@9#��~��@A#��~��@I#��~��@a���Ӝ$?i�ڐ���?�Unknown
�HostMatMul";training/SGD/gradients/gradients/dense/MatMul_grad/MatMul_1(17�A`�a�@97�A`�a�@A7�A`�a�@I7�A`�a�@a~�\��?i�7�Ef��?�Unknown
gHost_FusedMatMul"
�HostResourceApplyKerasMomentum"?training/SGD/SGD/update_dense/kernel/ResourceApplyKerasMomentum(1NbX9�a@9NbX9�a@ANbX9�a@INbX9�a@a�待s��>irQn���?�Unknown
�HostSoftmaxCrossEntropyWithLogits"1loss/dense_loss/softmax_cross_entropy_with_logits(1�~j�tB@9�~j�tB@A�~j�tB@I�~j�tB@aǷ-����>i��_b���?�Unknown
bHostSoftmax"
�HostResourceApplyKerasMomentum"=training/SGD/SGD/update_dense/bias/ResourceApplyKerasMomentum(1#��~j�(@9#��~j�(@A#��~j�(@I#��~j�(@aJUB�>i#g�'���?�Unknown
�HostAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1����x)'@9����x)'@A����x)'@I����x)'@aR�Z졵>if������?�Unknown
~HostReadVariableOp""block5_conv2/Conv2D/ReadVariableOp(1�� �r�%@9�� �r�%@A�� �r�%@I�� �r�%@apW4�R:�>i
�HostTile">training/SGD/gradients/gradients/loss/dense_loss/Sum_grad/Tile(1%��CK%@9%��CK%@A%��CK%@I%��CK%@aJ�K�>i�������?�Unknown
kHostArgMax"metrics/accuracy/ArgMax(1���K�"@9���K�"@A���K�"@I���K�"@a�|���>iui���?�Unknown
�HostConcatV2":loss/dense_loss/softmax_cross_entropy_with_logits/concat_1(1sh��|� @9sh��|� @Ash��|� @Ish��|� @a�ב��H�>i
0�
���?�Unknown
w HostReadVariableOp"dense/MatMul/ReadVariableOp(1�/�$� @9�/�$� @A�/�$� @I�/�$� @a9�U�ݮ>id�����?�Unknown
�!HostMul"[training/SGD/gradients/gradients/loss/dense_loss/softmax_cross_entropy_with_logits_grad/mul(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@az&�;ֈ�>i!y\����?�Unknown
g"HostCast"metrics/accuracy/Cast(1����xi@9����xi@A����xi@I����xi@a���Dx�>i!��x���?�Unknown
�#HostSlice"9loss/dense_loss/softmax_cross_entropy_with_logits/Slice_1(1���Mb@9���Mb@A���Mb@I���Mb@a��l,%�>i��1+���?�Unknown
�$HostReadVariableOp"&training/SGD/Identity_1/ReadVariableOp(1��Q��@9��Q��@A��Q��@I��Q��@ay�w����>i��a����?�Unknown
~%HostReadVariableOp""block1_conv2/Conv2D/ReadVariableOp(1�G�z@9�G�z@A�G�z@I�G�z@a'>>�l�>i3�)L���?�Unknown
d&HostSum"loss/dense_loss/Sum(1Zd;�O
'HostReadVariableOp"#block4_conv3/BiasAdd/ReadVariableOp(1ˡE���@9ˡE���@AˡE���@IˡE���@a�Q.��>i�98����?�Unknown
e(HostSum"metrics/accuracy/Sum(1������@9������@A������@I������@a��U,.m�>i�3���?�Unknown
)HostReadVariableOp"#block1_conv2/BiasAdd/ReadVariableOp(15^�I�@95^�I�@A5^�I�@I5^�I�@a=bP�D8�>i�k�V���?�Unknown
~*HostReadVariableOp""block3_conv3/Conv2D/ReadVariableOp(1/�$@9/�$@A/�$@I/�$@aC��ÿ�>ir���?�Unknown
+HostReadVariableOp"#block5_conv1/BiasAdd/ReadVariableOp(1-����@9-����@A-����@I-����@a�`�l@��>i������?�Unknown
�,HostBiasAddGrad"?training/SGD/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad(1�G�z.@9�G�z.@A�G�z.@I�G�z.@ad��;���>iy�˚���?�Unknown
~-HostReadVariableOp""block4_conv1/Conv2D/ReadVariableOp(1)\���(@9)\���(@A)\���(@I)\���(@aF����>i�6+����?�Unknown
m.HostArgMax"metrics/accuracy/ArgMax_1(1�����@9�����@A�����@I�����@a t�7�>i	:����?�Unknown
�/HostReadVariableOp"$training/SGD/Identity/ReadVariableOp(1�|?5^:@9�|?5^:@A�|?5^:@I�|?5^:@a���!�>i�W�����?�Unknown
x0HostReadVariableOp"dense/BiasAdd/ReadVariableOp(1V-2@9V-2@AV-2@IV-2@a���1{�>i�
�����?�Unknown
~1HostReadVariableOp""block2_conv2/Conv2D/ReadVariableOp(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a��Td$�>i+�����?�Unknown
2HostReadVariableOp"#block3_conv3/BiasAdd/ReadVariableOp(1\���(\@9\���(\@A\���(\@I\���(\@a@�}�J�>i�_�����?�Unknown
~3HostReadVariableOp""block5_conv1/Conv2D/ReadVariableOp(1+���@9+���@A+���@I+���@a��ȓƜ>iv�'����?�Unknown
~4HostReadVariableOp""block3_conv2/Conv2D/ReadVariableOp(1`��"��
5HostReadVariableOp"#block3_conv2/BiasAdd/ReadVariableOp(1�t�V
6HostReadVariableOp"#block2_conv2/BiasAdd/ReadVariableOp(1��ʡE
�7HostAssignAddVariableOp"$training/SGD/SGD/AssignAddVariableOp(1�rh��|@9�rh��|@A�rh��|@I�rh��|@aInx&X��>iG������?�Unknown
8HostReadVariableOp"#block5_conv3/BiasAdd/ReadVariableOp(1��� �r@9��� �r@A��� �r@I��� �r@a@��Ǒ�>i��p����?�Unknown
s9HostCast"!loss/dense_loss/num_elements/Cast(1�K7�A`@9�K7�A`@A�K7�A`@I�K7�A`@a�5Opx��>iO������?�Unknown
~:HostReadVariableOp""block5_conv3/Conv2D/ReadVariableOp(1��Q��
@9��Q��
@A��Q��
@I��Q��
@a�� �Ę>i�#R���?�Unknown
�;HostReadVariableOp"*metrics/accuracy/div_no_nan/ReadVariableOp(1�C�l��@9�C�l��@A�C�l��@I�C�l��@a�5@�U�>i�t����?�Unknown
<HostReadVariableOp"#block4_conv2/BiasAdd/ReadVariableOp(1%��C�@9%��C�@A%��C�@I%��C�@a�8	SZ�>i�G2����?�Unknown
i=HostEqual"metrics/accuracy/Equal(1Zd;�O@9Zd;�O@AZd;�O@IZd;�O@a��M�ᴖ>idW�y���?�Unknown
~>HostReadVariableOp""block2_conv1/Conv2D/ReadVariableOp(1�n���@9�n���@A�n���@I�n���@a�����8�>iiĝ+���?�Unknown
?HostReadVariableOp"#block2_conv1/BiasAdd/ReadVariableOp(1��C�l@9��C�l@A��C�l@I��C�l@a_* ���>i�K�����?�Unknown
~@HostReadVariableOp""block4_conv2/Conv2D/ReadVariableOp(1������@9������@A������@I������@a��>��>i�~����?�Unknown
AHostReadVariableOp"#block1_conv1/BiasAdd/ReadVariableOp(1�Zd;�@9�Zd;�@A�Zd;�@I�Zd;�@ac�h�L��>i]����?�Unknown
BHostReadVariableOp"#block5_conv2/BiasAdd/ReadVariableOp(1��"��~@9��"��~@A��"��~@I��"��~@aB�NF�>i�+����?�Unknown
~CHostReadVariableOp""block4_conv3/Conv2D/ReadVariableOp(1V-���@9V-���@AV-���@IV-���@a��di��>i"�.(���?�Unknown
~DHostReadVariableOp""block3_conv1/Conv2D/ReadVariableOp(1�K7�A`@9�K7�A`@A�K7�A`@I�K7�A`@a����:�>i������?�Unknown
�EHostDivNoNan"Ftraining/SGD/gradients/gradients/loss/dense_loss/value_grad/div_no_nan(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?a�s�ʼ�>i9(� ���?�Unknown
FHostReadVariableOp"#block3_conv1/BiasAdd/ReadVariableOp(1y�&1��?9y�&1��?Ay�&1��?Iy�&1��?a_��}VǊ>i0�����?�Unknown
~GHostReadVariableOp""block1_conv1/Conv2D/ReadVariableOp(1m������?9m������?Am������?Im������?a]Fʕ��>iY�����?�Unknown
�HHostAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1��S㥛�?9��S㥛�?A��S㥛�?I��S㥛�?a]�����>i�w�O���?�Unknown
kIHostDivNoNan"loss/dense_loss/value(1`��"���?9`��"���?A`��"���?I`��"���?a�W)�su�>i^Fʥ���?�Unknown
JHostReadVariableOp"#block4_conv1/BiasAdd/ReadVariableOp(1+�����?9+�����?A+�����?I+�����?a����q��>i������?�Unknown
qKHostDivNoNan"metrics/accuracy/div_no_nan(1�ʡE���?9�ʡE���?A�ʡE���?I�ʡE���?a�Q>8���>iv��I���?�Unknown
iLHostCast"metrics/accuracy/Cast_1(1V-����?9V-����?AV-����?IV-����?a��di��>i+�����?�Unknown
�MHostSum"Ktraining/SGD/gradients/gradients/loss/dense_loss/weighted_loss/Mul_grad/Sum(1���Q��?9���Q��?A���Q��?I���Q��?a��RX;>i�V����?�Unknown
�NHostReadVariableOp",metrics/accuracy/div_no_nan/ReadVariableOp_1(1��x�&1�?9��x�&1�?A��x�&1�?I��x�&1�?a�~�x�Tz>i      �?�Unknown*�J
kHost_FusedConv2D"block4_conv3/Relu(1��/!�A9��/!�AA��/!�AI��/!�Aa6�~}N�?i6�~}N�?�Unknown
kHost_FusedConv2D"block4_conv2/Relu(1���x�[A9���x�[AA���x�[AI���x�[Aah�|�ٿ?i�FJy��?�Unknown
kHost_FusedConv2D"block1_conv2/Relu(1#��~��
kHost_FusedConv2D"block2_conv2/Relu(1�O���QA9�O���QAA�O���QAI�O���QAaX�pɑ�?i�E�ѢG�?�Unknown
kHost_FusedConv2D"block3_conv2/Relu(1-�����
A9-�����
AA-�����
AI-�����
Aa�o�t.�?i�P!�i�?�Unknown
kHost_FusedConv2D"block3_conv3/Relu(1��~j��
A9��~j��
AA��~j��
AI��~j��
Aa�p�c(�?i�je���?�Unknown
kHost_FusedConv2D"block4_conv1/Relu(1�S㥱��@9�S㥱��@A�S㥱��@I�S㥱��@aV�}WȪ?iFF�J-[�?�Unknown
kHost_FusedConv2D"block2_conv1/Relu(1�O����@9�O����@A�O����@I�O����@a_T�0�?i8�L6��?�Unknown
k	Host_FusedConv2D"block3_conv1/Relu(1y�&1��@9y�&1��@Ay�&1��@Iy�&1��@a*����I�?i['��Ғ�?�Unknown
k
Host_FusedConv2D"block5_conv2/Relu(1V-��N�@9V-��N�@AV-��N�@IV-��N�@a�)R��ˢ?i�I
kHost_FusedConv2D"block5_conv1/Relu(1��x����@9��x����@A��x����@I��x����@aQ�шk�?i�RJ��?�Unknown
kHost_FusedConv2D"block5_conv3/Relu(1�ʡEʊ�@9�ʡEʊ�@A�ʡEʊ�@I�ʡEʊ�@ai���̴�?iy�����?�Unknown
h
kHost_FusedConv2D"block1_conv1/Relu(1`��"���@9`��"���@A`��"���@I`��"���@a��	ڝ>�?iR'X��>�?�Unknown
hHostMaxPool"block2_pool/MaxPool(1X9��ff�@9X9��ff�@AX9��ff�@IX9��ff�@a�T�k���?i���r��?�Unknown
hHostMaxPool"block3_pool/MaxPool(1����l/�@9����l/�@A����l/�@I����l/�@a|ع,�}y?iW`\n��?�Unknown
hHostMaxPool"block4_pool/MaxPool(1ףp=j��@9ףp=j��@Aףp=j��@Iףp=j��@aC��7?�h?i0	��G��?�Unknown
hHostMaxPool"block5_pool/MaxPool(1#��~��@9#��~��@A#��~��@I#��~��@a)�+ 0|E?i�����?�Unknown
�HostMatMul";training/SGD/gradients/gradients/dense/MatMul_grad/MatMul_1(17�A`�a�@97�A`�a�@A7�A`�a�@I7�A`�a�@aD�<�C�6?i�{��?�Unknown
gHost_FusedMatMul"
�HostResourceApplyKerasMomentum"?training/SGD/SGD/update_dense/kernel/ResourceApplyKerasMomentum(1NbX9�a@9NbX9�a@ANbX9�a@INbX9�a@a�KUP~?i���?���?�Unknown
�HostSoftmaxCrossEntropyWithLogits"1loss/dense_loss/softmax_cross_entropy_with_logits(1�~j�tB@9�~j�tB@A�~j�tB@I�~j�tB@a��Ҙ�>i
bHostSoftmax"
�HostResourceApplyKerasMomentum"=training/SGD/SGD/update_dense/bias/ResourceApplyKerasMomentum(1#��~j�(@9#��~j�(@A#��~j�(@I#��~j�(@aH�>��>i�F��#��?�Unknown
�HostAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1����x)'@9����x)'@A����x)'@I����x)'@a5�/�U��>i�8)�.��?�Unknown
~HostReadVariableOp""block5_conv2/Conv2D/ReadVariableOp(1�� �r�%@9�� �r�%@A�� �r�%@I�� �r�%@a��lփ�>i�#�|9��?�Unknown
�HostTile">training/SGD/gradients/gradients/loss/dense_loss/Sum_grad/Tile(1%��CK%@9%��CK%@A%��CK%@I%��CK%@a5�{�̺�>i��Q�C��?�Unknown
kHostArgMax"metrics/accuracy/ArgMax(1���K�"@9���K�"@A���K�"@I���K�"@a(��K�v�>i㾡M��?�Unknown
�HostConcatV2":loss/dense_loss/softmax_cross_entropy_with_logits/concat_1(1sh��|� @9sh��|� @Ash��|� @Ish��|� @a����M�>i���<U��?�Unknown
wHostReadVariableOp"dense/MatMul/ReadVariableOp(1�/�$� @9�/�$� @A�/�$� @I�/�$� @a1N�
�>i�ҕG]��?�Unknown
�HostMul"[training/SGD/gradients/gradients/loss/dense_loss/softmax_cross_entropy_with_logits_grad/mul(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@a�Ds<
��>i�a�d��?�Unknown
g HostCast"metrics/accuracy/Cast(1����xi@9����xi@A����xi@I����xi@ac����>i�(��k��?�Unknown
�!HostSlice"9loss/dense_loss/softmax_cross_entropy_with_logits/Slice_1(1���Mb@9���Mb@A���Mb@I���Mb@a��Z�4K�>i:ga�r��?�Unknown
�"HostReadVariableOp"&training/SGD/Identity_1/ReadVariableOp(1��Q��@9��Q��@A��Q��@I��Q��@ae�0���>id�I�y��?�Unknown
~#HostReadVariableOp""block1_conv2/Conv2D/ReadVariableOp(1�G�z@9�G�z@A�G�z@I�G�z@a���Tj�>i�'����?�Unknown
d$HostSum"loss/dense_loss/Sum(1Zd;�O
%HostReadVariableOp"#block4_conv3/BiasAdd/ReadVariableOp(1ˡE���@9ˡE���@AˡE���@IˡE���@az���o�>i6'�2���?�Unknown
e&HostSum"metrics/accuracy/Sum(1������@9������@A������@I������@aA���?�>i6-�B���?�Unknown
'HostReadVariableOp"#block1_conv2/BiasAdd/ReadVariableOp(15^�I�@95^�I�@A5^�I�@I5^�I�@a�T{���>i����?�Unknown
~(HostReadVariableOp""block3_conv3/Conv2D/ReadVariableOp(1/�$@9/�$@A/�$@I/�$@av�˪��>i�6�����?�Unknown
)HostReadVariableOp"#block5_conv1/BiasAdd/ReadVariableOp(1-����@9-����@A-����@I-����@a�)qFG�>i ��3���?�Unknown
�*HostBiasAddGrad"?training/SGD/gradients/gradients/dense/BiasAdd_grad/BiasAddGrad(1�G�z.@9�G�z.@A�G�z.@I�G�z.@a�����>i:Yx����?�Unknown
~+HostReadVariableOp""block4_conv1/Conv2D/ReadVariableOp(1)\���(@9)\���(@A)\���(@I)\���(@a�m�­�>io�����?�Unknown
m,HostArgMax"metrics/accuracy/ArgMax_1(1�����@9�����@A�����@I�����@a���0S�>i���`���?�Unknown
�-HostReadVariableOp"$training/SGD/Identity/ReadVariableOp(1�|?5^:@9�|?5^:@A�|?5^:@I�|?5^:@a�ݮ9}��>i�
����?�Unknown
x.HostReadVariableOp"dense/BiasAdd/ReadVariableOp(1V-2@9V-2@AV-2@IV-2@awY����>i��k����?�Unknown
~/HostReadVariableOp""block2_conv2/Conv2D/ReadVariableOp(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a���:�>i'з��?�Unknown
0HostReadVariableOp"#block3_conv3/BiasAdd/ReadVariableOp(1\���(\@9\���(\@A\���(\@I\���(\@af#�h��>i�����?�Unknown
~1HostReadVariableOp""block5_conv1/Conv2D/ReadVariableOp(1+���@9+���@A+���@I+���@a#"����>i���`���?�Unknown
~2HostReadVariableOp""block3_conv2/Conv2D/ReadVariableOp(1`��"��
3HostReadVariableOp"#block3_conv2/BiasAdd/ReadVariableOp(1�t�V
4HostReadVariableOp"#block2_conv2/BiasAdd/ReadVariableOp(1��ʡE
�5HostAssignAddVariableOp"$training/SGD/SGD/AssignAddVariableOp(1�rh��|@9�rh��|@A�rh��|@I�rh��|@a�Q����>i��ԟ���?�Unknown
6HostReadVariableOp"#block5_conv3/BiasAdd/ReadVariableOp(1��� �r@9��� �r@A��� �r@I��� �r@a�7&����>i�
s7HostCast"!loss/dense_loss/num_elements/Cast(1�K7�A`@9�K7�A`@A�K7�A`@I�K7�A`@a�j݇��>i3	�j���?�Unknown
~8HostReadVariableOp""block5_conv3/Conv2D/ReadVariableOp(1��Q��
@9��Q��
@A��Q��
@I��Q��
@a�`1�3ѹ>i9�����?�Unknown
�9HostReadVariableOp"*metrics/accuracy/div_no_nan/ReadVariableOp(1�C�l��@9�C�l��@A�C�l��@I�C�l��@aGhR�>i<�O����?�Unknown
:HostReadVariableOp"#block4_conv2/BiasAdd/ReadVariableOp(1%��C�@9%��C�@A%��C�@I%��C�@a�;n��>i�����?�Unknown
i;HostEqual"metrics/accuracy/Equal(1Zd;�O@9Zd;�O@AZd;�O@IZd;�O@a�K
~<HostReadVariableOp""block2_conv1/Conv2D/ReadVariableOp(1�n���@9�n���@A�n���@I�n���@a�!�gW)�>i@^p����?�Unknown
=HostReadVariableOp"#block2_conv1/BiasAdd/ReadVariableOp(1��C�l@9��C�l@A��C�l@I��C�l@a��~I�Ͷ>ip�$`���?�Unknown
~>HostReadVariableOp""block4_conv2/Conv2D/ReadVariableOp(1������@9������@A������@I������@as���F �>ic^- ���?�Unknown
?HostReadVariableOp"#block1_conv1/BiasAdd/ReadVariableOp(1�Zd;�@9�Zd;�@A�Zd;�@I�Zd;�@aZ�DgjX�>iL�:����?�Unknown
@HostReadVariableOp"#block5_conv2/BiasAdd/ReadVariableOp(1��"��~@9��"��~@A��"��~@I��"��~@a��ا~�>iG�j����?�Unknown
~AHostReadVariableOp""block4_conv3/Conv2D/ReadVariableOp(1V-���@9V-���@AV-���@IV-���@a�oz�u�>i�b(����?�Unknown
~BHostReadVariableOp""block3_conv1/Conv2D/ReadVariableOp(1�K7�A`@9�K7�A`@A�K7�A`@I�K7�A`@a|��_�>iYSt���?�Unknown
�CHostDivNoNan"Ftraining/SGD/gradients/gradients/loss/dense_loss/value_grad/div_no_nan(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?as�����>i��d���?�Unknown
DHostReadVariableOp"#block3_conv1/BiasAdd/ReadVariableOp(1y�&1��?9y�&1��?Ay�&1��?Iy�&1��?aO6���>i������?�Unknown
~EHostReadVariableOp""block1_conv1/Conv2D/ReadVariableOp(1m������?9m������?Am������?Im������?ak�-�>iN�_w���?�Unknown
�FHostAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1��S㥛�?9��S㥛�?A��S㥛�?I��S㥛�?axw申��>i� �����?�Unknown
kGHostDivNoNan"loss/dense_loss/value(1`��"���?9`��"���?A`��"���?I`��"���?a��:�]�>iIԊ\���?�Unknown
HHostReadVariableOp"#block4_conv1/BiasAdd/ReadVariableOp(1+�����?9+�����?A+�����?I+�����?a.S���b�>i�-�����?�Unknown
qIHostDivNoNan"metrics/accuracy/div_no_nan(1�ʡE���?9�ʡE���?A�ʡE���?I�ʡE���?a�e��^�>i溣���?�Unknown
iJHostCast"metrics/accuracy/Cast_1(1V-����?9V-����?AV-����?IV-����?a�oz�u�>i>� ���?�Unknown
�KHostSum"Ktraining/SGD/gradients/gradients/loss/dense_loss/weighted_loss/Mul_grad/Sum(1���Q��?9���Q��?A���Q��?I���Q��?a��6�F�>i��p$���?�Unknown
�LHostReadVariableOp",metrics/accuracy/div_no_nan/ReadVariableOp_1(1��x�&1�?9��x�&1�?A��x�&1�?I��x�&1�?aA^w
�q�>i�������?�Unknown2Nvidia GPU (Turing)