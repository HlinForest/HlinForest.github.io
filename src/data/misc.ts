import type { ImageMetadata } from 'astro';
import panoramaSrc from '../assets/misc/tokyo-skyscanner-reference.png';
import tokyo1 from '../assets/misc/tokyo-blue-hour-01.webp';
import tokyo2 from '../assets/misc/tokyo-blue-hour-02.webp';
import tokyo3 from '../assets/misc/tokyo-blue-hour-03.webp';
import tokyo4 from '../assets/misc/tokyo-blue-hour-04.webp';
import lake from '../assets/misc/garden-alan-lake.png';
import rain from '../assets/misc/garden-alan-rain-field.png';
import hydrangea from '../assets/misc/garden-alan-hydrangea.png';
import lily from '../assets/misc/garden-alan-lily.png';
import khalil from '../assets/misc/spike-impressionist-portrait.png';
import yusheng from '../assets/misc/zhang-yusheng-impressionist-portrait.png';

export interface Credit { text: string; href?: string; linkLabel?: string; license?: { label: string; href: string } }
export interface MiscItem { src: ImageMetadata; alt: string; label: string; group: string; credit: Credit; quote?: string[] }
const tokyoCredit: Credit = { text: 'Photograph · selected reference', href: 'https://www.sina.cn/news/detail/5155104072140071.html', linkLabel: 'Text reference · _等待夏夜吹成烈焰 / 新浪' };
const gardenCredit: Credit = { text: 'Photography · Alan D. Haller', href: 'https://alan-d-haller.info/travel_jp/thegardenofwords/', linkLabel: 'Source · 雨中的新宿御苑' };
export const panorama: MiscItem = { src: panoramaSrc, alt: '东京天际线、东京塔与远处富士山的宽幅城市景观', label: 'Tokyo skyline', group: 'Photography', credit: { text: 'Favorite image · Skyscanner reference', href: 'https://www.skyscanner.com.hk/news/bc/stories-of-our-favourite-travel-photos', linkLabel: 'Visual reference · Skyscanner travel stories' } };
export const photos: MiscItem[] = [
  { src: tokyo1, alt: '蓝调时刻的东京铁塔与城市天际线', label: 'Tokyo · Blue hour 01', group: 'Tokyo Blue Hour', credit: tokyoCredit },
  { src: tokyo2, alt: '蓝调时刻中被云层环抱的东京高楼', label: 'Tokyo · Blue hour 02', group: 'Tokyo Blue Hour', credit: tokyoCredit },
  { src: tokyo3, alt: '彩霞映照下的东京湾与城市天际线', label: 'Tokyo · Blue hour 03', group: 'Tokyo Blue Hour', credit: tokyoCredit },
  { src: tokyo4, alt: '晚霞后的东京城市天际线', label: 'Tokyo · Blue hour 04', group: 'Tokyo Blue Hour', credit: tokyoCredit },
  { src: lake, alt: '雨中新宿御苑被树木环抱的湖面', label: '言叶之庭 · 雨天湖面', group: '言叶之庭', credit: gardenCredit },
  { src: rain, alt: '雨幕中的新宿御苑草地与远处树木', label: '言叶之庭 · 雨幕草地', group: '言叶之庭', credit: gardenCredit },
  { src: hydrangea, alt: '雨水覆盖的蓝紫色绣球花与深绿叶片', label: '言叶之庭 · 雨中绣球', group: '言叶之庭', credit: gardenCredit },
  { src: lily, alt: '雨中的白色百合花与水珠', label: '言叶之庭 · 雨中百合', group: '言叶之庭', credit: gardenCredit },
];
export const musicians: MiscItem[] = [
  { src: khalil, alt: '由方大同 2007 年照片转换的印象派肖像画', label: '方大同', group: '方大同', quote: ['也没有一时澎湃悸动，却在深秋细水长流。'], credit: { text: 'AI-transformed', href: 'https://commons.wikimedia.org/wiki/File:KhalilFong2007.jpg', linkLabel: 'After Khalil Fong, 2007 · Verity', license: { label: 'CC BY-SA 2.0', href: 'https://creativecommons.org/licenses/by-sa/2.0/' } } },
  { src: yusheng, alt: '张雨生年轻时期的印象派肖像画', label: '张雨生', group: '张雨生', quote: ['我的胸口鼓鼓吹胀', '欢乐幸福的遐想'], credit: { text: 'AI-generated impressionist portrait study' } },
];
