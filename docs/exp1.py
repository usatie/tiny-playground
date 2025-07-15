from tinygrad.helpers import Timing
from tinygrad import Tensor
from tinygrad import dtypes
from tinygrad.nn.datasets import mnist
import numpy as np

class Linear:
  def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
    self.weight = getattr(Tensor, initialization)(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    return x.linear(self.weight.transpose(), self.bias)
class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leaky_relu()
    x = self.l2(x)
    return x

net = TinyNet()

print("---tinygrad indexing---")
X_train, Y_train, X_test, Y_test = mnist()
with Timing("Time: "):
  avg_acc = 0
  num_steps = 512
  for step in range(num_steps):
    # random sample a batch
    samp = Tensor.randint(64, high=X_test.shape[0])
    batch = X_test[samp].reshape(64, -1)  # flatten the images to (64, 784)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1)
    avg_acc = avg_acc + (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc.item() / num_steps}")

# If I change the num_steps to 512, it raises an error:
# (If it's 511, it works fine)
# DEBUG=4 python exp1.py
"""
__kernel void r_16_4n1(__global float* data0, __global bool* data1, __global int* data2, __global int* data3, __global int* data4, __global int* data5, __global int* data6, __global int* data7, __global int* data8, __global int* data9, __global int* data10, __global int* data11, __global int* data12, __global int* data13, __global int* data14, __global int* data15, __global int* data16, __global int* data17, __global int* data18, __global int* data19, __global int* data20, __global int* data21, __global int* data22, __global int* data23, __global int* data24, __global int* data25, __global int* data26, __global int* data27, __global int* data28, __global int* data29, __global int* data30, __global int* data31, __global int* data32, __global int* data33, __global int* data34, __global int* data35, __global int* data36, __global int* data37, __global int* data38, __global int* data39, __global int* data40, __global int* data41, __global int* data42, __global int* data43, __global int* data44, __global int* data45, __global int* data46, __global int* data47, __global int* data48, __global int* data49, __global int* data50, __global int* data51, __global int* data52, __global int* data53, __global int* data54, __global int* data55, __global int* data56, __global int* data57, __global int* data58, __global int* data59, __global int* data60, __global int* data61, __global int* data62, __global int* data63, __global int* data64, __global int* data65, __global int* data66, __global int* data67, __global int* data68, __global int* data69, __global int* data70, __global int* data71, __global int* data72, __global int* data73, __global int* data74, __global int* data75, __global int* data76, __global int* data77, __global int* data78, __global int* data79, __global int* data80, __global int* data81, __global int* data82, __global int* data83, __global int* data84, __global int* data85, __global int* data86, __global int* data87, __global int* data88, __global int* data89, __global int* data90, __global int* data91, __global int* data92, __global int* data93, __global int* data94, __global int* data95, __global int* data96, __global int* data97, __global int* data98, __global int* data99, __global int* data100, __global int* data101, __global int* data102, __global int* data103, __global int* data104, __global int* data105, __global int* data106, __global int* data107, __global int* data108, __global int* data109, __global int* data110, __global int* data111, __global int* data112, __global int* data113, __global int* data114, __global int* data115, __global int* data116, __global int* data117, __global int* data118, __global int* data119, __global int* data120, __global int* data121, __global int* data122, __global int* data123, __global int* data124, __global int* data125, __global int* data126, __global int* data127, __global int* data128, __global int* data129, __global int* data130, __global int* data131, __global int* data132, __global int* data133, __global int* data134, __global int* data135, __global int* data136, __global int* data137, __global int* data138, __global int* data139, __global int* data140, __global int* data141, __global int* data142, __global int* data143, __global int* data144, __global int* data145, __global int* data146, __global int* data147, __global int* data148, __global int* data149, __global int* data150, __global int* data151, __global int* data152, __global int* data153, __global int* data154, __global int* data155, __global int* data156, __global int* data157, __global int* data158, __global int* data159, __global int* data160, __global int* data161, __global int* data162, __global int* data163, __global int* data164, __global int* data165, __global int* data166, __global int* data167, __global int* data168, __global int* data169, __global int* data170, __global int* data171, __global int* data172, __global int* data173, __global int* data174, __global int* data175, __global int* data176, __global int* data177, __global int* data178, __global int* data179, __global int* data180, __global int* data181, __global int* data182, __global int* data183, __global int* data184, __global int* data185, __global int* data186, __global int* data187, __global int* data188, __global int* data189, __global int* data190, __global int* data191, __global int* data192, __global int* data193, __global int* data194, __global int* data195, __global int* data196, __global int* data197, __global int* data198, __global int* data199, __global int* data200, __global int* data201, __global int* data202, __global int* data203, __global int* data204, __global int* data205, __global int* data206, __global int* data207, __global int* data208, __global int* data209, __global int* data210, __global int* data211, __global int* data212, __global int* data213, __global int* data214, __global int* data215, __global int* data216, __global int* data217, __global int* data218, __global int* data219, __global int* data220, __global int* data221, __global int* data222, __global int* data223, __global int* data224, __global int* data225, __global int* data226, __global int* data227, __global int* data228, __global int* data229, __global int* data230, __global int* data231, __global int* data232, __global int* data233, __global int* data234, __global int* data235, __global int* data236, __global int* data237, __global int* data238, __global int* data239, __global int* data240, __global int* data241, __global int* data242, __global int* data243, __global int* data244, __global int* data245, __global int* data246, __global int* data247, __global int* data248, __global int* data249, __global int* data250, __global int* data251, __global int* data252, __global int* data253, __global int* data254, __global int* data255, __global int* data256, __global int* data257, __global int* data258, __global int* data259, __global int* data260, __global int* data261, __global int* data262, __global int* data263, __global int* data264, __global int* data265, __global int* data266, __global int* data267, __global int* data268, __global int* data269, __global int* data270, __global int* data271, __global int* data272, __global int* data273, __global int* data274, __global int* data275, __global int* data276, __global int* data277, __global int* data278, __global int* data279, __global int* data280, __global int* data281, __global int* data282, __global int* data283, __global int* data284, __global int* data285, __global int* data286, __global int* data287, __global int* data288, __global int* data289, __global int* data290, __global int* data291, __global int* data292, __global int* data293, __global int* data294, __global int* data295, __global int* data296, __global int* data297, __global int* data298, __global int* data299, __global int* data300, __global int* data301, __global int* data302, __global int* data303, __global int* data304, __global int* data305, __global int* data306, __global int* data307, __global int* data308, __global int* data309, __global int* data310, __global int* data311, __global int* data312, __global int* data313, __global int* data314, __global int* data315, __global int* data316, __global int* data317, __global int* data318, __global int* data319, __global int* data320, __global int* data321, __global int* data322, __global int* data323, __global int* data324, __global int* data325, __global int* data326, __global int* data327, __global int* data328, __global int* data329, __global int* data330, __global int* data331, __global int* data332, __global int* data333, __global int* data334, __global int* data335, __global int* data336, __global int* data337, __global int* data338, __global int* data339, __global int* data340, __global int* data341, __global int* data342, __global int* data343, __global int* data344, __global int* data345, __global int* data346, __global int* data347, __global int* data348, __global int* data349, __global int* data350, __global int* data351, __global int* data352, __global int* data353, __global int* data354, __global int* data355, __global int* data356, __global int* data357, __global int* data358, __global int* data359, __global int* data360, __global int* data361, __global int* data362, __global int* data363, __global int* data364, __global int* data365, __global int* data366, __global int* data367, __global int* data368, __global int* data369, __global int* data370, __global int* data371, __global int* data372, __global int* data373, __global int* data374, __global int* data375, __global int* data376, __global int* data377, __global int* data378, __global int* data379, __global int* data380, __global int* data381, __global int* data382, __global int* data383, __global int* data384, __global int* data385, __global int* data386, __global int* data387, __global int* data388, __global int* data389, __global int* data390, __global int* data391, __global int* data392, __global int* data393, __global int* data394, __global int* data395, __global int* data396, __global int* data397, __global int* data398, __global int* data399, __global int* data400, __global int* data401, __global int* data402, __global int* data403, __global int* data404, __global int* data405, __global int* data406, __global int* data407, __global int* data408, __global int* data409, __global int* data410, __global int* data411, __global int* data412, __global int* data413, __global int* data414, __global int* data415, __global int* data416, __global int* data417, __global int* data418, __global int* data419, __global int* data420, __global int* data421, __global int* data422, __global int* data423, __global int* data424, __global int* data425, __global int* data426, __global int* data427, __global int* data428, __global int* data429, __global int* data430, __global int* data431, __global int* data432, __global int* data433, __global int* data434, __global int* data435, __global int* data436, __global int* data437, __global int* data438, __global int* data439, __global int* data440, __global int* data441, __global int* data442, __global int* data443, __global int* data444, __global int* data445, __global int* data446, __global int* data447, __global int* data448, __global int* data449, __global int* data450, __global int* data451, __global int* data452, __global int* data453, __global int* data454, __global int* data455, __global int* data456, __global int* data457, __global int* data458, __global int* data459, __global int* data460, __global int* data461, __global int* data462, __global int* data463, __global int* data464, __global int* data465, __global int* data466, __global int* data467, __global int* data468, __global int* data469, __global int* data470, __global int* data471, __global int* data472, __global int* data473, __global int* data474, __global int* data475, __global int* data476, __global int* data477, __global int* data478, __global int* data479, __global int* data480, __global int* data481, __global int* data482, __global int* data483, __global int* data484, __global int* data485, __global int* data486, __global int* data487, __global int* data488, __global int* data489, __global int* data490, __global int* data491, __global int* data492, __global int* data493, __global int* data494, __global int* data495, __global int* data496, __global int* data497, __global int* data498, __global int* data499, __global int* data500, __global int* data501, __global int* data502, __global int* data503, __global int* data504, __global int* data505, __global int* data506, __global int* data507, __global int* data508, __global int* data509, __global int* data510, __global int* data511, __global int* data512) {
  __attribute__ ((aligned (16))) __local int temp0[16];
  int lidx0 = get_local_id(0); /* 16 */
  int val0 = *(data2+0);
  int val1 = *(data3+0);
  int val2 = *(data4+0);
  int val3 = *(data5+0);
  int val4 = *(data6+0);
  int val5 = *(data7+0);
  int val6 = *(data8+0);
  int val7 = *(data9+0);
  int val8 = *(data10+0);
  int val9 = *(data11+0);
  int val10 = *(data12+0);
  int val11 = *(data13+0);
  int val12 = *(data14+0);
  int val13 = *(data15+0);
  int val14 = *(data16+0);
  int val15 = *(data17+0);
  int val16 = *(data18+0);
  int val17 = *(data19+0);
  int val18 = *(data20+0);
  int val19 = *(data21+0);
  int val20 = *(data22+0);
  int val21 = *(data23+0);
  int val22 = *(data24+0);
  int val23 = *(data25+0);
  int val24 = *(data26+0);
  int val25 = *(data27+0);
  int val26 = *(data28+0);
  int val27 = *(data29+0);
  int val28 = *(data30+0);
  int val29 = *(data31+0);
  int val30 = *(data32+0);
  int val31 = *(data33+0);
  int val32 = *(data34+0);
  int val33 = *(data35+0);
  int val34 = *(data36+0);
  int val35 = *(data37+0);
  int val36 = *(data38+0);
  int val37 = *(data39+0);
  int val38 = *(data40+0);
  int val39 = *(data41+0);
  int val40 = *(data42+0);
  int val41 = *(data43+0);
  int val42 = *(data44+0);
  int val43 = *(data45+0);
  int val44 = *(data46+0);
  int val45 = *(data47+0);
  int val46 = *(data48+0);
  int val47 = *(data49+0);
  int val48 = *(data50+0);
  int val49 = *(data51+0);
  int val50 = *(data52+0);
  int val51 = *(data53+0);
  int val52 = *(data54+0);
  int val53 = *(data55+0);
  int val54 = *(data56+0);
  int val55 = *(data57+0);
  int val56 = *(data58+0);
  int val57 = *(data59+0);
  int val58 = *(data60+0);
  int val59 = *(data61+0);
  int val60 = *(data62+0);
  int val61 = *(data63+0);
  int val62 = *(data64+0);
  int val63 = *(data65+0);
  int val64 = *(data66+0);
  int val65 = *(data67+0);
  int val66 = *(data68+0);
  int val67 = *(data69+0);
  int val68 = *(data70+0);
  int val69 = *(data71+0);
  int val70 = *(data72+0);
  int val71 = *(data73+0);
  int val72 = *(data74+0);
  int val73 = *(data75+0);
  int val74 = *(data76+0);
  int val75 = *(data77+0);
  int val76 = *(data78+0);
  int val77 = *(data79+0);
  int val78 = *(data80+0);
  int val79 = *(data81+0);
  int val80 = *(data82+0);
  int val81 = *(data83+0);
  int val82 = *(data84+0);
  int val83 = *(data85+0);
  int val84 = *(data86+0);
  int val85 = *(data87+0);
  int val86 = *(data88+0);
  int val87 = *(data89+0);
  int val88 = *(data90+0);
  int val89 = *(data91+0);
  int val90 = *(data92+0);
  int val91 = *(data93+0);
  int val92 = *(data94+0);
  int val93 = *(data95+0);
  int val94 = *(data96+0);
  int val95 = *(data97+0);
  int val96 = *(data98+0);
  int val97 = *(data99+0);
  int val98 = *(data100+0);
  int val99 = *(data101+0);
  int val100 = *(data102+0);
  int val101 = *(data103+0);
  int val102 = *(data104+0);
  int val103 = *(data105+0);
  int val104 = *(data106+0);
  int val105 = *(data107+0);
  int val106 = *(data108+0);
  int val107 = *(data109+0);
  int val108 = *(data110+0);
  int val109 = *(data111+0);
  int val110 = *(data112+0);
  int val111 = *(data113+0);
  int val112 = *(data114+0);
  int val113 = *(data115+0);
  int val114 = *(data116+0);
  int val115 = *(data117+0);
  int val116 = *(data118+0);
  int val117 = *(data119+0);
  int val118 = *(data120+0);
  int val119 = *(data121+0);
  int val120 = *(data122+0);
  int val121 = *(data123+0);
  int val122 = *(data124+0);
  int val123 = *(data125+0);
  int val124 = *(data126+0);
  int val125 = *(data127+0);
  int val126 = *(data128+0);
  int val127 = *(data129+0);
  int val128 = *(data130+0);
  int val129 = *(data131+0);
  int val130 = *(data132+0);
  int val131 = *(data133+0);
  int val132 = *(data134+0);
  int val133 = *(data135+0);
  int val134 = *(data136+0);
  int val135 = *(data137+0);
  int val136 = *(data138+0);
  int val137 = *(data139+0);
  int val138 = *(data140+0);
  int val139 = *(data141+0);
  int val140 = *(data142+0);
  int val141 = *(data143+0);
  int val142 = *(data144+0);
  int val143 = *(data145+0);
  int val144 = *(data146+0);
  int val145 = *(data147+0);
  int val146 = *(data148+0);
  int val147 = *(data149+0);
  int val148 = *(data150+0);
  int val149 = *(data151+0);
  int val150 = *(data152+0);
  int val151 = *(data153+0);
  int val152 = *(data154+0);
  int val153 = *(data155+0);
  int val154 = *(data156+0);
  int val155 = *(data157+0);
  int val156 = *(data158+0);
  int val157 = *(data159+0);
  int val158 = *(data160+0);
  int val159 = *(data161+0);
  int val160 = *(data162+0);
  int val161 = *(data163+0);
  int val162 = *(data164+0);
  int val163 = *(data165+0);
  int val164 = *(data166+0);
  int val165 = *(data167+0);
  int val166 = *(data168+0);
  int val167 = *(data169+0);
  int val168 = *(data170+0);
  int val169 = *(data171+0);
  int val170 = *(data172+0);
  int val171 = *(data173+0);
  int val172 = *(data174+0);
  int val173 = *(data175+0);
  int val174 = *(data176+0);
  int val175 = *(data177+0);
  int val176 = *(data178+0);
  int val177 = *(data179+0);
  int val178 = *(data180+0);
  int val179 = *(data181+0);
  int val180 = *(data182+0);
  int val181 = *(data183+0);
  int val182 = *(data184+0);
  int val183 = *(data185+0);
  int val184 = *(data186+0);
  int val185 = *(data187+0);
  int val186 = *(data188+0);
  int val187 = *(data189+0);
  int val188 = *(data190+0);
  int val189 = *(data191+0);
  int val190 = *(data192+0);
  int val191 = *(data193+0);
  int val192 = *(data194+0);
  int val193 = *(data195+0);
  int val194 = *(data196+0);
  int val195 = *(data197+0);
  int val196 = *(data198+0);
  int val197 = *(data199+0);
  int val198 = *(data200+0);
  int val199 = *(data201+0);
  int val200 = *(data202+0);
  int val201 = *(data203+0);
  int val202 = *(data204+0);
  int val203 = *(data205+0);
  int val204 = *(data206+0);
  int val205 = *(data207+0);
  int val206 = *(data208+0);
  int val207 = *(data209+0);
  int val208 = *(data210+0);
  int val209 = *(data211+0);
  int val210 = *(data212+0);
  int val211 = *(data213+0);
  int val212 = *(data214+0);
  int val213 = *(data215+0);
  int val214 = *(data216+0);
  int val215 = *(data217+0);
  int val216 = *(data218+0);
  int val217 = *(data219+0);
  int val218 = *(data220+0);
  int val219 = *(data221+0);
  int val220 = *(data222+0);
  int val221 = *(data223+0);
  int val222 = *(data224+0);
  int val223 = *(data225+0);
  int val224 = *(data226+0);
  int val225 = *(data227+0);
  int val226 = *(data228+0);
  int val227 = *(data229+0);
  int val228 = *(data230+0);
  int val229 = *(data231+0);
  int val230 = *(data232+0);
  int val231 = *(data233+0);
  int val232 = *(data234+0);
  int val233 = *(data235+0);
  int val234 = *(data236+0);
  int val235 = *(data237+0);
  int val236 = *(data238+0);
  int val237 = *(data239+0);
  int val238 = *(data240+0);
  int val239 = *(data241+0);
  int val240 = *(data242+0);
  int val241 = *(data243+0);
  int val242 = *(data244+0);
  int val243 = *(data245+0);
  int val244 = *(data246+0);
  int val245 = *(data247+0);
  int val246 = *(data248+0);
  int val247 = *(data249+0);
  int val248 = *(data250+0);
  int val249 = *(data251+0);
  int val250 = *(data252+0);
  int val251 = *(data253+0);
  int val252 = *(data254+0);
  int val253 = *(data255+0);
  int val254 = *(data256+0);
  int val255 = *(data257+0);
  int val256 = *(data258+0);
  int val257 = *(data259+0);
  int val258 = *(data260+0);
  int val259 = *(data261+0);
  int val260 = *(data262+0);
  int val261 = *(data263+0);
  int val262 = *(data264+0);
  int val263 = *(data265+0);
  int val264 = *(data266+0);
  int val265 = *(data267+0);
  int val266 = *(data268+0);
  int val267 = *(data269+0);
  int val268 = *(data270+0);
  int val269 = *(data271+0);
  int val270 = *(data272+0);
  int val271 = *(data273+0);
  int val272 = *(data274+0);
  int val273 = *(data275+0);
  int val274 = *(data276+0);
  int val275 = *(data277+0);
  int val276 = *(data278+0);
  int val277 = *(data279+0);
  int val278 = *(data280+0);
  int val279 = *(data281+0);
  int val280 = *(data282+0);
  int val281 = *(data283+0);
  int val282 = *(data284+0);
  int val283 = *(data285+0);
  int val284 = *(data286+0);
  int val285 = *(data287+0);
  int val286 = *(data288+0);
  int val287 = *(data289+0);
  int val288 = *(data290+0);
  int val289 = *(data291+0);
  int val290 = *(data292+0);
  int val291 = *(data293+0);
  int val292 = *(data294+0);
  int val293 = *(data295+0);
  int val294 = *(data296+0);
  int val295 = *(data297+0);
  int val296 = *(data298+0);
  int val297 = *(data299+0);
  int val298 = *(data300+0);
  int val299 = *(data301+0);
  int val300 = *(data302+0);
  int val301 = *(data303+0);
  int val302 = *(data304+0);
  int val303 = *(data305+0);
  int val304 = *(data306+0);
  int val305 = *(data307+0);
  int val306 = *(data308+0);
  int val307 = *(data309+0);
  int val308 = *(data310+0);
  int val309 = *(data311+0);
  int val310 = *(data312+0);
  int val311 = *(data313+0);
  int val312 = *(data314+0);
  int val313 = *(data315+0);
  int val314 = *(data316+0);
  int val315 = *(data317+0);
  int val316 = *(data318+0);
  int val317 = *(data319+0);
  int val318 = *(data320+0);
  int val319 = *(data321+0);
  int val320 = *(data322+0);
  int val321 = *(data323+0);
  int val322 = *(data324+0);
  int val323 = *(data325+0);
  int val324 = *(data326+0);
  int val325 = *(data327+0);
  int val326 = *(data328+0);
  int val327 = *(data329+0);
  int val328 = *(data330+0);
  int val329 = *(data331+0);
  int val330 = *(data332+0);
  int val331 = *(data333+0);
  int val332 = *(data334+0);
  int val333 = *(data335+0);
  int val334 = *(data336+0);
  int val335 = *(data337+0);
  int val336 = *(data338+0);
  int val337 = *(data339+0);
  int val338 = *(data340+0);
  int val339 = *(data341+0);
  int val340 = *(data342+0);
  int val341 = *(data343+0);
  int val342 = *(data344+0);
  int val343 = *(data345+0);
  int val344 = *(data346+0);
  int val345 = *(data347+0);
  int val346 = *(data348+0);
  int val347 = *(data349+0);
  int val348 = *(data350+0);
  int val349 = *(data351+0);
  int val350 = *(data352+0);
  int val351 = *(data353+0);
  int val352 = *(data354+0);
  int val353 = *(data355+0);
  int val354 = *(data356+0);
  int val355 = *(data357+0);
  int val356 = *(data358+0);
  int val357 = *(data359+0);
  int val358 = *(data360+0);
  int val359 = *(data361+0);
  int val360 = *(data362+0);
  int val361 = *(data363+0);
  int val362 = *(data364+0);
  int val363 = *(data365+0);
  int val364 = *(data366+0);
  int val365 = *(data367+0);
  int val366 = *(data368+0);
  int val367 = *(data369+0);
  int val368 = *(data370+0);
  int val369 = *(data371+0);
  int val370 = *(data372+0);
  int val371 = *(data373+0);
  int val372 = *(data374+0);
  int val373 = *(data375+0);
  int val374 = *(data376+0);
  int val375 = *(data377+0);
  int val376 = *(data378+0);
  int val377 = *(data379+0);
  int val378 = *(data380+0);
  int val379 = *(data381+0);
  int val380 = *(data382+0);
  int val381 = *(data383+0);
  int val382 = *(data384+0);
  int val383 = *(data385+0);
  int val384 = *(data386+0);
  int val385 = *(data387+0);
  int val386 = *(data388+0);
  int val387 = *(data389+0);
  int val388 = *(data390+0);
  int val389 = *(data391+0);
  int val390 = *(data392+0);
  int val391 = *(data393+0);
  int val392 = *(data394+0);
  int val393 = *(data395+0);
  int val394 = *(data396+0);
  int val395 = *(data397+0);
  int val396 = *(data398+0);
  int val397 = *(data399+0);
  int val398 = *(data400+0);
  int val399 = *(data401+0);
  int val400 = *(data402+0);
  int val401 = *(data403+0);
  int val402 = *(data404+0);
  int val403 = *(data405+0);
  int val404 = *(data406+0);
  int val405 = *(data407+0);
  int val406 = *(data408+0);
  int val407 = *(data409+0);
  int val408 = *(data410+0);
  int val409 = *(data411+0);
  int val410 = *(data412+0);
  int val411 = *(data413+0);
  int val412 = *(data414+0);
  int val413 = *(data415+0);
  int val414 = *(data416+0);
  int val415 = *(data417+0);
  int val416 = *(data418+0);
  int val417 = *(data419+0);
  int val418 = *(data420+0);
  int val419 = *(data421+0);
  int val420 = *(data422+0);
  int val421 = *(data423+0);
  int val422 = *(data424+0);
  int val423 = *(data425+0);
  int val424 = *(data426+0);
  int val425 = *(data427+0);
  int val426 = *(data428+0);
  int val427 = *(data429+0);
  int val428 = *(data430+0);
  int val429 = *(data431+0);
  int val430 = *(data432+0);
  int val431 = *(data433+0);
  int val432 = *(data434+0);
  int val433 = *(data435+0);
  int val434 = *(data436+0);
  int val435 = *(data437+0);
  int val436 = *(data438+0);
  int val437 = *(data439+0);
  int val438 = *(data440+0);
  int val439 = *(data441+0);
  int val440 = *(data442+0);
  int val441 = *(data443+0);
  int val442 = *(data444+0);
  int val443 = *(data445+0);
  int val444 = *(data446+0);
  int val445 = *(data447+0);
  int val446 = *(data448+0);
  int val447 = *(data449+0);
  int val448 = *(data450+0);
  int val449 = *(data451+0);
  int val450 = *(data452+0);
  int val451 = *(data453+0);
  int val452 = *(data454+0);
  int val453 = *(data455+0);
  int val454 = *(data456+0);
  int val455 = *(data457+0);
  int val456 = *(data458+0);
  int val457 = *(data459+0);
  int val458 = *(data460+0);
  int val459 = *(data461+0);
  int val460 = *(data462+0);
  int val461 = *(data463+0);
  int val462 = *(data464+0);
  int val463 = *(data465+0);
  int val464 = *(data466+0);
  int val465 = *(data467+0);
  int val466 = *(data468+0);
  int val467 = *(data469+0);
  int val468 = *(data470+0);
  int val469 = *(data471+0);
  int val470 = *(data472+0);
  int val471 = *(data473+0);
  int val472 = *(data474+0);
  int val473 = *(data475+0);
  int val474 = *(data476+0);
  int val475 = *(data477+0);
  int val476 = *(data478+0);
  int val477 = *(data479+0);
  int val478 = *(data480+0);
  int val479 = *(data481+0);
  int val480 = *(data482+0);
  int val481 = *(data483+0);
  int val482 = *(data484+0);
  int val483 = *(data485+0);
  int val484 = *(data486+0);
  int val485 = *(data487+0);
  int val486 = *(data488+0);
  int val487 = *(data489+0);
  int val488 = *(data490+0);
  int val489 = *(data491+0);
  int val490 = *(data492+0);
  int val491 = *(data493+0);
  int val492 = *(data494+0);
  int val493 = *(data495+0);
  int val494 = *(data496+0);
  int val495 = *(data497+0);
  int val496 = *(data498+0);
  int val497 = *(data499+0);
  int val498 = *(data500+0);
  int val499 = *(data501+0);
  int val500 = *(data502+0);
  int val501 = *(data503+0);
  int val502 = *(data504+0);
  int val503 = *(data505+0);
  int val504 = *(data506+0);
  int val505 = *(data507+0);
  int val506 = *(data508+0);
  int val507 = *(data509+0);
  int val508 = *(data510+0);
  int val509 = *(data511+0);
  int val510 = *(data512+0);
  int acc0 = 0;
  for (int ridx1 = 0; ridx1 < 4; ridx1++) {
    bool val511 = *(data1+((lidx0<<2)+ridx1));
    acc0 = (acc0+((int)(val511)));
  }
  *(temp0+lidx0) = acc0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((((bool)(lidx0))!=1)) {
    int acc1 = 0;
    for (int ridx1000 = 0; ridx1000 < 16; ridx1000++) {
      int val512 = *(temp0+ridx1000);
      acc1 = (acc1+val512);
    }
    *(data0+0) = ((((float)(acc1))*0.015625f)+(((float)(val0))*0.015625f)+(((float)(val1))*0.015625f)+(((float)(val2))*0.015625f)+(((float)(val3))*0.015625f)+(((float)(val4))*0.015625f)+(((float)(val5))*0.015625f)+(((float)(val6))*0.015625f)+(((float)(val7))*0.015625f)+(((float)(val8))*0.015625f)+(((float)(val9))*0.015625f)+(((float)(val10))*0.015625f)+(((float)(val11))*0.015625f)+(((float)(val12))*0.015625f)+(((float)(val13))*0.015625f)+(((float)(val14))*0.015625f)+(((float)(val15))*0.015625f)+(((float)(val16))*0.015625f)+(((float)(val17))*0.015625f)+(((float)(val18))*0.015625f)+(((float)(val19))*0.015625f)+(((float)(val20))*0.015625f)+(((float)(val21))*0.015625f)+(((float)(val22))*0.015625f)+(((float)(val23))*0.015625f)+(((float)(val24))*0.015625f)+(((float)(val25))*0.015625f)+(((float)(val26))*0.015625f)+(((float)(val27))*0.015625f)+(((float)(val28))*0.015625f)+(((float)(val29))*0.015625f)+(((float)(val30))*0.015625f)+(((float)(val31))*0.015625f)+(((float)(val32))*0.015625f)+(((float)(val33))*0.015625f)+(((float)(val34))*0.015625f)+(((float)(val35))*0.015625f)+(((float)(val36))*0.015625f)+(((float)(val37))*0.015625f)+(((float)(val38))*0.015625f)+(((float)(val39))*0.015625f)+(((float)(val40))*0.015625f)+(((float)(val41))*0.015625f)+(((float)(val42))*0.015625f)+(((float)(val43))*0.015625f)+(((float)(val44))*0.015625f)+(((float)(val45))*0.015625f)+(((float)(val46))*0.015625f)+(((float)(val47))*0.015625f)+(((float)(val48))*0.015625f)+(((float)(val49))*0.015625f)+(((float)(val50))*0.015625f)+(((float)(val51))*0.015625f)+(((float)(val52))*0.015625f)+(((float)(val53))*0.015625f)+(((float)(val54))*0.015625f)+(((float)(val55))*0.015625f)+(((float)(val56))*0.015625f)+(((float)(val57))*0.015625f)+(((float)(val58))*0.015625f)+(((float)(val59))*0.015625f)+(((float)(val60))*0.015625f)+(((float)(val61))*0.015625f)+(((float)(val62))*0.015625f)+(((float)(val63))*0.015625f)+(((float)(val64))*0.015625f)+(((float)(val65))*0.015625f)+(((float)(val66))*0.015625f)+(((float)(val67))*0.015625f)+(((float)(val68))*0.015625f)+(((float)(val69))*0.015625f)+(((float)(val70))*0.015625f)+(((float)(val71))*0.015625f)+(((float)(val72))*0.015625f)+(((float)(val73))*0.015625f)+(((float)(val74))*0.015625f)+(((float)(val75))*0.015625f)+(((float)(val76))*0.015625f)+(((float)(val77))*0.015625f)+(((float)(val78))*0.015625f)+(((float)(val79))*0.015625f)+(((float)(val80))*0.015625f)+(((float)(val81))*0.015625f)+(((float)(val82))*0.015625f)+(((float)(val83))*0.015625f)+(((float)(val84))*0.015625f)+(((float)(val85))*0.015625f)+(((float)(val86))*0.015625f)+(((float)(val87))*0.015625f)+(((float)(val88))*0.015625f)+(((float)(val89))*0.015625f)+(((float)(val90))*0.015625f)+(((float)(val91))*0.015625f)+(((float)(val92))*0.015625f)+(((float)(val93))*0.015625f)+(((float)(val94))*0.015625f)+(((float)(val95))*0.015625f)+(((float)(val96))*0.015625f)+(((float)(val97))*0.015625f)+(((float)(val98))*0.015625f)+(((float)(val99))*0.015625f)+(((float)(val100))*0.015625f)+(((float)(val101))*0.015625f)+(((float)(val102))*0.015625f)+(((float)(val103))*0.015625f)+(((float)(val104))*0.015625f)+(((float)(val105))*0.015625f)+(((float)(val106))*0.015625f)+(((float)(val107))*0.015625f)+(((float)(val108))*0.015625f)+(((float)(val109))*0.015625f)+(((float)(val110))*0.015625f)+(((float)(val111))*0.015625f)+(((float)(val112))*0.015625f)+(((float)(val113))*0.015625f)+(((float)(val114))*0.015625f)+(((float)(val115))*0.015625f)+(((float)(val116))*0.015625f)+(((float)(val117))*0.015625f)+(((float)(val118))*0.015625f)+(((float)(val119))*0.015625f)+(((float)(val120))*0.015625f)+(((float)(val121))*0.015625f)+(((float)(val122))*0.015625f)+(((float)(val123))*0.015625f)+(((float)(val124))*0.015625f)+(((float)(val125))*0.015625f)+(((float)(val126))*0.015625f)+(((float)(val127))*0.015625f)+(((float)(val128))*0.015625f)+(((float)(val129))*0.015625f)+(((float)(val130))*0.015625f)+(((float)(val131))*0.015625f)+(((float)(val132))*0.015625f)+(((float)(val133))*0.015625f)+(((float)(val134))*0.015625f)+(((float)(val135))*0.015625f)+(((float)(val136))*0.015625f)+(((float)(val137))*0.015625f)+(((float)(val138))*0.015625f)+(((float)(val139))*0.015625f)+(((float)(val140))*0.015625f)+(((float)(val141))*0.015625f)+(((float)(val142))*0.015625f)+(((float)(val143))*0.015625f)+(((float)(val144))*0.015625f)+(((float)(val145))*0.015625f)+(((float)(val146))*0.015625f)+(((float)(val147))*0.015625f)+(((float)(val148))*0.015625f)+(((float)(val149))*0.015625f)+(((float)(val150))*0.015625f)+(((float)(val151))*0.015625f)+(((float)(val152))*0.015625f)+(((float)(val153))*0.015625f)+(((float)(val154))*0.015625f)+(((float)(val155))*0.015625f)+(((float)(val156))*0.015625f)+(((float)(val157))*0.015625f)+(((float)(val158))*0.015625f)+(((float)(val159))*0.015625f)+(((float)(val160))*0.015625f)+(((float)(val161))*0.015625f)+(((float)(val162))*0.015625f)+(((float)(val163))*0.015625f)+(((float)(val164))*0.015625f)+(((float)(val165))*0.015625f)+(((float)(val166))*0.015625f)+(((float)(val167))*0.015625f)+(((float)(val168))*0.015625f)+(((float)(val169))*0.015625f)+(((float)(val170))*0.015625f)+(((float)(val171))*0.015625f)+(((float)(val172))*0.015625f)+(((float)(val173))*0.015625f)+(((float)(val174))*0.015625f)+(((float)(val175))*0.015625f)+(((float)(val176))*0.015625f)+(((float)(val177))*0.015625f)+(((float)(val178))*0.015625f)+(((float)(val179))*0.015625f)+(((float)(val180))*0.015625f)+(((float)(val181))*0.015625f)+(((float)(val182))*0.015625f)+(((float)(val183))*0.015625f)+(((float)(val184))*0.015625f)+(((float)(val185))*0.015625f)+(((float)(val186))*0.015625f)+(((float)(val187))*0.015625f)+(((float)(val188))*0.015625f)+(((float)(val189))*0.015625f)+(((float)(val190))*0.015625f)+(((float)(val191))*0.015625f)+(((float)(val192))*0.015625f)+(((float)(val193))*0.015625f)+(((float)(val194))*0.015625f)+(((float)(val195))*0.015625f)+(((float)(val196))*0.015625f)+(((float)(val197))*0.015625f)+(((float)(val198))*0.015625f)+(((float)(val199))*0.015625f)+(((float)(val200))*0.015625f)+(((float)(val201))*0.015625f)+(((float)(val202))*0.015625f)+(((float)(val203))*0.015625f)+(((float)(val204))*0.015625f)+(((float)(val205))*0.015625f)+(((float)(val206))*0.015625f)+(((float)(val207))*0.015625f)+(((float)(val208))*0.015625f)+(((float)(val209))*0.015625f)+(((float)(val210))*0.015625f)+(((float)(val211))*0.015625f)+(((float)(val212))*0.015625f)+(((float)(val213))*0.015625f)+(((float)(val214))*0.015625f)+(((float)(val215))*0.015625f)+(((float)(val216))*0.015625f)+(((float)(val217))*0.015625f)+(((float)(val218))*0.015625f)+(((float)(val219))*0.015625f)+(((float)(val220))*0.015625f)+(((float)(val221))*0.015625f)+(((float)(val222))*0.015625f)+(((float)(val223))*0.015625f)+(((float)(val224))*0.015625f)+(((float)(val225))*0.015625f)+(((float)(val226))*0.015625f)+(((float)(val227))*0.015625f)+(((float)(val228))*0.015625f)+(((float)(val229))*0.015625f)+(((float)(val230))*0.015625f)+(((float)(val231))*0.015625f)+(((float)(val232))*0.015625f)+(((float)(val233))*0.015625f)+(((float)(val234))*0.015625f)+(((float)(val235))*0.015625f)+(((float)(val236))*0.015625f)+(((float)(val237))*0.015625f)+(((float)(val238))*0.015625f)+(((float)(val239))*0.015625f)+(((float)(val240))*0.015625f)+(((float)(val241))*0.015625f)+(((float)(val242))*0.015625f)+(((float)(val243))*0.015625f)+(((float)(val244))*0.015625f)+(((float)(val245))*0.015625f)+(((float)(val246))*0.015625f)+(((float)(val247))*0.015625f)+(((float)(val248))*0.015625f)+(((float)(val249))*0.015625f)+(((float)(val250))*0.015625f)+(((float)(val251))*0.015625f)+(((float)(val252))*0.015625f)+(((float)(val253))*0.015625f)+(((float)(val254))*0.015625f)+(((float)(val255))*0.015625f)+(((float)(val256))*0.015625f)+(((float)(val257))*0.015625f)+(((float)(val258))*0.015625f)+(((float)(val259))*0.015625f)+(((float)(val260))*0.015625f)+(((float)(val261))*0.015625f)+(((float)(val262))*0.015625f)+(((float)(val263))*0.015625f)+(((float)(val264))*0.015625f)+(((float)(val265))*0.015625f)+(((float)(val266))*0.015625f)+(((float)(val267))*0.015625f)+(((float)(val268))*0.015625f)+(((float)(val269))*0.015625f)+(((float)(val270))*0.015625f)+(((float)(val271))*0.015625f)+(((float)(val272))*0.015625f)+(((float)(val273))*0.015625f)+(((float)(val274))*0.015625f)+(((float)(val275))*0.015625f)+(((float)(val276))*0.015625f)+(((float)(val277))*0.015625f)+(((float)(val278))*0.015625f)+(((float)(val279))*0.015625f)+(((float)(val280))*0.015625f)+(((float)(val281))*0.015625f)+(((float)(val282))*0.015625f)+(((float)(val283))*0.015625f)+(((float)(val284))*0.015625f)+(((float)(val285))*0.015625f)+(((float)(val286))*0.015625f)+(((float)(val287))*0.015625f)+(((float)(val288))*0.015625f)+(((float)(val289))*0.015625f)+(((float)(val290))*0.015625f)+(((float)(val291))*0.015625f)+(((float)(val292))*0.015625f)+(((float)(val293))*0.015625f)+(((float)(val294))*0.015625f)+(((float)(val295))*0.015625f)+(((float)(val296))*0.015625f)+(((float)(val297))*0.015625f)+(((float)(val298))*0.015625f)+(((float)(val299))*0.015625f)+(((float)(val300))*0.015625f)+(((float)(val301))*0.015625f)+(((float)(val302))*0.015625f)+(((float)(val303))*0.015625f)+(((float)(val304))*0.015625f)+(((float)(val305))*0.015625f)+(((float)(val306))*0.015625f)+(((float)(val307))*0.015625f)+(((float)(val308))*0.015625f)+(((float)(val309))*0.015625f)+(((float)(val310))*0.015625f)+(((float)(val311))*0.015625f)+(((float)(val312))*0.015625f)+(((float)(val313))*0.015625f)+(((float)(val314))*0.015625f)+(((float)(val315))*0.015625f)+(((float)(val316))*0.015625f)+(((float)(val317))*0.015625f)+(((float)(val318))*0.015625f)+(((float)(val319))*0.015625f)+(((float)(val320))*0.015625f)+(((float)(val321))*0.015625f)+(((float)(val322))*0.015625f)+(((float)(val323))*0.015625f)+(((float)(val324))*0.015625f)+(((float)(val325))*0.015625f)+(((float)(val326))*0.015625f)+(((float)(val327))*0.015625f)+(((float)(val328))*0.015625f)+(((float)(val329))*0.015625f)+(((float)(val330))*0.015625f)+(((float)(val331))*0.015625f)+(((float)(val332))*0.015625f)+(((float)(val333))*0.015625f)+(((float)(val334))*0.015625f)+(((float)(val335))*0.015625f)+(((float)(val336))*0.015625f)+(((float)(val337))*0.015625f)+(((float)(val338))*0.015625f)+(((float)(val339))*0.015625f)+(((float)(val340))*0.015625f)+(((float)(val341))*0.015625f)+(((float)(val342))*0.015625f)+(((float)(val343))*0.015625f)+(((float)(val344))*0.015625f)+(((float)(val345))*0.015625f)+(((float)(val346))*0.015625f)+(((float)(val347))*0.015625f)+(((float)(val348))*0.015625f)+(((float)(val349))*0.015625f)+(((float)(val350))*0.015625f)+(((float)(val351))*0.015625f)+(((float)(val352))*0.015625f)+(((float)(val353))*0.015625f)+(((float)(val354))*0.015625f)+(((float)(val355))*0.015625f)+(((float)(val356))*0.015625f)+(((float)(val357))*0.015625f)+(((float)(val358))*0.015625f)+(((float)(val359))*0.015625f)+(((float)(val360))*0.015625f)+(((float)(val361))*0.015625f)+(((float)(val362))*0.015625f)+(((float)(val363))*0.015625f)+(((float)(val364))*0.015625f)+(((float)(val365))*0.015625f)+(((float)(val366))*0.015625f)+(((float)(val367))*0.015625f)+(((float)(val368))*0.015625f)+(((float)(val369))*0.015625f)+(((float)(val370))*0.015625f)+(((float)(val371))*0.015625f)+(((float)(val372))*0.015625f)+(((float)(val373))*0.015625f)+(((float)(val374))*0.015625f)+(((float)(val375))*0.015625f)+(((float)(val376))*0.015625f)+(((float)(val377))*0.015625f)+(((float)(val378))*0.015625f)+(((float)(val379))*0.015625f)+(((float)(val380))*0.015625f)+(((float)(val381))*0.015625f)+(((float)(val382))*0.015625f)+(((float)(val383))*0.015625f)+(((float)(val384))*0.015625f)+(((float)(val385))*0.015625f)+(((float)(val386))*0.015625f)+(((float)(val387))*0.015625f)+(((float)(val388))*0.015625f)+(((float)(val389))*0.015625f)+(((float)(val390))*0.015625f)+(((float)(val391))*0.015625f)+(((float)(val392))*0.015625f)+(((float)(val393))*0.015625f)+(((float)(val394))*0.015625f)+(((float)(val395))*0.015625f)+(((float)(val396))*0.015625f)+(((float)(val397))*0.015625f)+(((float)(val398))*0.015625f)+(((float)(val399))*0.015625f)+(((float)(val400))*0.015625f)+(((float)(val401))*0.015625f)+(((float)(val402))*0.015625f)+(((float)(val403))*0.015625f)+(((float)(val404))*0.015625f)+(((float)(val405))*0.015625f)+(((float)(val406))*0.015625f)+(((float)(val407))*0.015625f)+(((float)(val408))*0.015625f)+(((float)(val409))*0.015625f)+(((float)(val410))*0.015625f)+(((float)(val411))*0.015625f)+(((float)(val412))*0.015625f)+(((float)(val413))*0.015625f)+(((float)(val414))*0.015625f)+(((float)(val415))*0.015625f)+(((float)(val416))*0.015625f)+(((float)(val417))*0.015625f)+(((float)(val418))*0.015625f)+(((float)(val419))*0.015625f)+(((float)(val420))*0.015625f)+(((float)(val421))*0.015625f)+(((float)(val422))*0.015625f)+(((float)(val423))*0.015625f)+(((float)(val424))*0.015625f)+(((float)(val425))*0.015625f)+(((float)(val426))*0.015625f)+(((float)(val427))*0.015625f)+(((float)(val428))*0.015625f)+(((float)(val429))*0.015625f)+(((float)(val430))*0.015625f)+(((float)(val431))*0.015625f)+(((float)(val432))*0.015625f)+(((float)(val433))*0.015625f)+(((float)(val434))*0.015625f)+(((float)(val435))*0.015625f)+(((float)(val436))*0.015625f)+(((float)(val437))*0.015625f)+(((float)(val438))*0.015625f)+(((float)(val439))*0.015625f)+(((float)(val440))*0.015625f)+(((float)(val441))*0.015625f)+(((float)(val442))*0.015625f)+(((float)(val443))*0.015625f)+(((float)(val444))*0.015625f)+(((float)(val445))*0.015625f)+(((float)(val446))*0.015625f)+(((float)(val447))*0.015625f)+(((float)(val448))*0.015625f)+(((float)(val449))*0.015625f)+(((float)(val450))*0.015625f)+(((float)(val451))*0.015625f)+(((float)(val452))*0.015625f)+(((float)(val453))*0.015625f)+(((float)(val454))*0.015625f)+(((float)(val455))*0.015625f)+(((float)(val456))*0.015625f)+(((float)(val457))*0.015625f)+(((float)(val458))*0.015625f)+(((float)(val459))*0.015625f)+(((float)(val460))*0.015625f)+(((float)(val461))*0.015625f)+(((float)(val462))*0.015625f)+(((float)(val463))*0.015625f)+(((float)(val464))*0.015625f)+(((float)(val465))*0.015625f)+(((float)(val466))*0.015625f)+(((float)(val467))*0.015625f)+(((float)(val468))*0.015625f)+(((float)(val469))*0.015625f)+(((float)(val470))*0.015625f)+(((float)(val471))*0.015625f)+(((float)(val472))*0.015625f)+(((float)(val473))*0.015625f)+(((float)(val474))*0.015625f)+(((float)(val475))*0.015625f)+(((float)(val476))*0.015625f)+(((float)(val477))*0.015625f)+(((float)(val478))*0.015625f)+(((float)(val479))*0.015625f)+(((float)(val480))*0.015625f)+(((float)(val481))*0.015625f)+(((float)(val482))*0.015625f)+(((float)(val483))*0.015625f)+(((float)(val484))*0.015625f)+(((float)(val485))*0.015625f)+(((float)(val486))*0.015625f)+(((float)(val487))*0.015625f)+(((float)(val488))*0.015625f)+(((float)(val489))*0.015625f)+(((float)(val490))*0.015625f)+(((float)(val491))*0.015625f)+(((float)(val492))*0.015625f)+(((float)(val493))*0.015625f)+(((float)(val494))*0.015625f)+(((float)(val495))*0.015625f)+(((float)(val496))*0.015625f)+(((float)(val497))*0.015625f)+(((float)(val498))*0.015625f)+(((float)(val499))*0.015625f)+(((float)(val500))*0.015625f)+(((float)(val501))*0.015625f)+(((float)(val502))*0.015625f)+(((float)(val503))*0.015625f)+(((float)(val504))*0.015625f)+(((float)(val505))*0.015625f)+(((float)(val506))*0.015625f)+(((float)(val507))*0.015625f)+(((float)(val508))*0.015625f)+(((float)(val509))*0.015625f)+(((float)(val510))*0.015625f));
  }
}
error lowering Ops.SINK
tensor operations:
(__add__, mean, __radd__)
Time: 9788.96 ms
Traceback (most recent call last):
  File "/Users/shunusami/workspace/tiny/playground/docs/exp1.py", line 45, in <module>
    print(f"Test Accuracy: {avg_acc.item() / num_steps}")
                            ^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4385, in _wrapper
    ret = fn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 328, in item
    return self.data()[(0,) * len(self.shape)]
           ^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
    if _METADATA.get() is not None: return fn(*args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 316, in data
    return self._buffer().as_typed_buffer(self.shape)
           ^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
    if _METADATA.get() is not None: return fn(*args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 302, in _buffer
    def _buffer(self) -> Buffer: return cast(Buffer, self.cast(self.dtype.base).contiguous().to("CPU").realize().uop.base.buffer)
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 4360, in _wrapper
    if _METADATA.get() is not None: return fn(*args, **kwargs)
                                           ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/tensor.py", line 269, in realize
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 192, in run_schedule
    for si, ei in lower_schedule(schedule):
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 185, in lower_schedule
    raise e
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 179, in lower_schedule
    try: yield (si, lower_schedule_item(si))
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 174, in lower_schedule_item
    return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata, si.fixedvars)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/uop/ops.py", line 730, in rewrite
    if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
             ^^^^^^^^^^^^^^^
  File "<string>", line 3, in compiled_match
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 167, in <lambda>
    (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 135, in get_runner
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/engine/realize.py", line 68, in __init__
    self._prg = Device[p.device].runtime(p.function_name, self.lib) if prg is None else prg
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/runtime/ops_gpu.py", line 41, in __init__
    self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), status := ctypes.c_int32()), status)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/runtime/ops_gpu.py", line 15, in checked
    def checked(ret, status): return (check(status.value), ret)[1]
                                      ^^^^^^^^^^^^^^^^^^^
  File "/Users/shunusami/workspace/tiny/tinygrad/tinygrad/runtime/ops_gpu.py", line 14, in check
    if status != 0: raise RuntimeError(f"OpenCL Error {status}: {cl_errors.get(status, 'Unknown error')}")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: OpenCL Error -48: CL_INVALID_KERNEL
"""
