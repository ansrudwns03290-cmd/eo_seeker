# 사용법: ./servo_jog.sh <채널>   (a: -5, d: +5, c: 중앙, q: 종료+출력 OFF)
CH=$1; REG=$((0x06 + 4*CH)); T=307; STEP=5
set_tick(){ i2cset -y 1 0x40 $REG 0x00 0x00 $((T & 0xFF)) $((T >> 8)) i; echo "ch$CH tick=$T"; }
set_tick
while read -rsn1 k; do
  case $k in
    a) T=$((T-STEP));;
    d) T=$((T+STEP));;
    c) T=307;;
    q) i2cset -y 1 0x40 $((REG+3)) 0x10; exit;;
  esac
  ((T<102)) && T=102; ((T>512)) && T=512   # 0.5~2.5ms 절대 한계
  set_tick
done
