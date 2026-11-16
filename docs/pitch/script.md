# What to say: the model part

About 185 words: close to 85 seconds at a calm pace (130 words a minute). Leaving out the parts
marked (can cut) brings it to about 75 seconds. Each English line is one breath. The Chinese line
under it says the same thing.

Numbers are said as "42 of 44", never as a percentage.

## Script

**Slide: the module diagram**

1. Our part is the model. For each corn field it answers one question: silage or grain?
   > 我们这部分是模型。它对每一块玉米地回答一个问题：青贮还是籽粒？

2. First it checks the evidence. Did the satellite see the harvest?
   > 第一步是检查证据：卫星有没有拍到收割？

3. If clouds hid the harvest, we do not guess. The field is "not judged".
   > 如果云挡住了收割，我们不猜。这块地记为"不判断"。

4. Then a small model decides. It uses six numbers.
   > 然后由一个小模型做判断。它只用六个数。

5. The key one is moisture just before harvest. Silage is cut wet. Grain is left to dry.
   > 最关键的一个是收割前的含水量。青贮是趁湿割的，籽粒玉米要等它变干。

**Slide: the labelled fields (fig3)**

6. We trained only on European fields. The model never saw a US field.
   > 我们只用欧洲的地块训练。模型从没见过美国的地。

7. Then we tested it on the organisers' labelled fields. It got 42 of 44 right.
   > 然后我们在主办方带标签的地块上测试。44 个里对了 42 个。

8. Always saying "grain" gets 32. A much bigger model got 37. (second sentence can cut)
   > 全部猜"籽粒"能对 32 个。一个大得多的模型对了 37 个。（第二句可删）

**Slide: counties and the state (fig5, state_2025)**

9. Last, we add the fields up to county and state acres.
   > 最后，我们把地块加起来，得到县和州的面积。

10. Here we must be honest. Our first county method failed its test in Maryland.
    > 这里必须说实话：我们第一种县级方法在马里兰没有通过检验。

11. So we went back to a plain count of the clearly seen fields.
    > 所以我们退回到最简单的办法：只数那些看清了收割的地块。

12. On two new counties in Michigan, the plain count passed its test.
    > 在密歇根两个新的县上，这个简单计数通过了检验。

13. But for Maryland in 2025 we read low: 22,700 acres of silage. USDA reports 40,000.
    > 但是对 2025 年的马里兰，我们的数偏低：青贮 22,700 英亩，而美国农业部的数字是 4 万。

14. We show every miss next to our numbers. (can cut)
    > 每一处偏差，我们都写在数字旁边。（可删）

How to say the numbers: 22,700 is "twenty-two thousand seven hundred"; 40,000 is "forty thousand".

**Still to come:** Pennsylvania and Wisconsin 2025. If they are logged in time, one more line
goes after line 13, in the same form.

## Five likely questions

Two sentences each. Do not add to them; if the judge wants more, point to the limits page.

**1. Your model never saw US data. Why should it work here?**
It looks at how wet the crop is when it is cut, not at the calendar date, and that is the same in Europe and here. On the organisers' labelled fields it got 42 of 44 right with no US training.
> 它看的是收割时作物有多湿，而不是日历上的日期，这一点欧洲和这里是一样的。在主办方带标签的地块上，它没有用任何美国数据训练，44 个对了 42 个。

**2. Why such a small model? Did you try deep learning or a pretrained model?**
Yes, we tried bigger models and a pretrained satellite model. In a new country they lose more accuracy than the small one, and on the labelled fields the best big model got 39 of 44 against 42.
> 试过，更大的模型和预训练的卫星模型都试了。换一个国家时，它们掉的精度比小模型多；在带标签的地块上，最好的大模型是 44 对 39，小模型是 42。

**3. How accurate are your county and state acres?**
Not good enough yet: the plain count passed its test on two Michigan counties, but for Maryland in 2025 it gives 22,700 acres of silage where USDA reports 40,000. Treat our acres as a first estimate that reads low on silage, not as a checked number.
> 还不够好：简单计数在密歇根两个县通过了检验，但对 2025 年的马里兰，它给出青贮 22,700 英亩，而美国农业部是 4 万。请把我们的面积看作一个青贮偏低的初步估计，而不是已经核实的数字。

**4. What do you do about clouds?**
If the last image before the cut and the first image after it are more than 20 days apart, the field is not judged. The silage share is measured only on the clearly seen fields, and we show how much of the corn that is: 62 percent in Maryland 2025.
> 如果收割前最后一张影像和收割后第一张影像相隔超过 20 天，这块地就不判断。青贮比例只在看清了的地块上计算，并且我们报告这些地块占玉米面积的多少：2025 年马里兰是 62%。

**5. Did you tune the model on the 44 labelled fields? Is 44 enough?**
No: the model was frozen on European data first, and every look at those fields is in a log. The set was scored three times, each time under a rule written beforehand, and 44 is small: one field is 2.3 points, so 42 and 43 would mean the same.
> 没有：模型先在欧洲数据上定型，对这些地块的每一次查看都记在日志里。这组地块一共打了三次分，每次都按事先写好的规则；44 个很少，一块地就是 2.3 个百分点，所以 42 和 43 没有实质区别。

## Words to keep the same everywhere

| Say | Not | 中文 |
|---|---|---|
| clearly seen / partly seen / not judged | clear tier, wide tier, abstain | 看清了 / 部分看到 / 不判断 |
| the small model | logistic regression, eu2_logistic_v1 | 小模型（六个特征的逻辑回归） |
| the big model | LightGBM, GBM | 大模型（135 个特征的提升树） |
| the first method (for county totals) | local threshold | 第一种县级方法（本地阈值） |
| the plain count | counted clear share | 简单计数（看清地块中判为青贮的面积比例） |
| 42 of 44 | 95 percent | 44 个里对 42 个 |
| reads low | is wrong, is validated | 偏低 |
