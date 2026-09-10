/** About Project and Evaluation pages. */

import { useAsync } from '../hooks'
import { api } from '../services/api'

/* -------------------------------------------------------------------------- */
/* About                                                                      */
/* -------------------------------------------------------------------------- */

export function AboutPage() {
  const stats = useAsync(() => api.statistics(), [])
  const backends = useAsync(() => api.pipeline(), [])
  const streams = useAsync(() => api.streams(), [])

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <header>
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
          เกี่ยวกับโครงงาน
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400" lang="th">
          Thai News Intelligence Dashboard · โครงงานรายวิชาการประมวลผลภาษาธรรมชาติ
        </p>
      </header>

      <Section title="ปัญหา (Problem)">
        <p lang="th">
          ข่าวภาษาไทยและปฏิกิริยาของผู้ชมเกิดขึ้นจำนวนมากในแต่ละวัน
          โดยเฉพาะในแชทสดของช่องข่าวบน YouTube ซึ่งมีข้อความหลายพันข้อความต่อรายการ
          การอ่านทั้งหมดด้วยมือเพื่อสรุปว่าผู้ชมกำลังพูดถึงเรื่องใดและมีความรู้สึกอย่างไร
          เป็นไปไม่ได้ในทางปฏิบัติ
        </p>
      </Section>

      <Section title="วัตถุประสงค์ (Objective)">
        <ul className="list-inside list-disc space-y-1" lang="th">
          <li>เก็บรวบรวมข้อความแชทสดภาษาไทยจากช่องข่าวบน YouTube โดยอัตโนมัติ</li>
          <li>จำแนกหัวข้อข่าวออกเป็น 15 หมวดหมู่</li>
          <li>วิเคราะห์ความรู้สึกเป็น 3 ระดับ พร้อมค่าความมั่นใจ</li>
          <li>สกัดคำสำคัญและสร้างบทสรุปโดยอัตโนมัติ</li>
          <li>นำเสนอผลผ่านแดชบอร์ดที่อธิบายที่มาของผลลัพธ์ได้</li>
        </ul>
      </Section>

      <Section title="เทคนิค NLP ที่ใช้ (NLP Techniques)">
        <dl className="space-y-3" lang="th">
          <Technique
            name="การตัดคำ (Word Tokenization)"
            detail="PyThaiNLP newmm (dictionary-based maximum matching) เนื่องจากภาษาไทยไม่มีการเว้นวรรคระหว่างคำ"
          />
          <Technique
            name="การทำความสะอาดข้อความ"
            detail="ลบ URL, HTML, อักขระซ้ำ และปรับรูปอักขระไทย โดยรักษาอีโมจิและคำหัวเราะ (555) ไว้เพราะสื่อความรู้สึก"
          />
          <Technique
            name="การตัดคำหยุด (Stopword Removal)"
            detail="ใช้คลังคำหยุดของ PyThaiNLP ร่วมกับคำฟิลเลอร์ในแชท แต่ปกป้องคำปฏิเสธ (ไม่, อย่า) ไว้เสมอ เพราะการตัดออกจะทำให้ผลวิเคราะห์ความรู้สึกกลับด้าน"
          />
          <Technique
            name="การจำแนกหัวข้อ (Topic Classification)"
            detail="คลังศัพท์บ่งชี้หมวด (gazetteer) รองรับคำประสม เป็นค่าฐาน และสลับเป็น TF-IDF + Linear model ได้เมื่อฝึกโมเดลแล้ว"
          />
          <Technique
            name="การวิเคราะห์ความรู้สึก (Sentiment Analysis)"
            detail="คลังคำบ่งชี้อารมณ์พร้อมการจัดการคำปฏิเสธและคำขยาย ระดับข้อความ แล้วสรุปเป็นภาพรวมของช่วงแชท"
          />
          <Technique
            name="การสกัดคำสำคัญ (Keyword Extraction)"
            detail="TF-IDF โดยคำนวณ IDF จากคลังข้อมูลจริงที่เก็บได้"
          />
          <Technique
            name="การสรุปความ (Summarization)"
            detail="แบบสกัดประโยค (extractive) ด้วยความเป็นศูนย์กลาง TF-IDF ร่วมกับน้ำหนักประโยคต้น ทำงานได้โดยไม่ต้องใช้ API ภายนอก"
          />
          <Technique
            name="การรู้จำชื่อเฉพาะ (NER)"
            detail="กฎและคลังคำ ครอบคลุม 77 จังหวัด คำนำหน้าชื่อบุคคล องค์กร วันที่ จำนวนเงิน และปริมาณ"
          />
        </dl>
      </Section>

      <Section title="สถาปัตยกรรมระบบ (System Architecture)">
        <pre className="scroll-thin overflow-x-auto rounded-xl bg-slate-50 p-4 text-[11px] leading-relaxed text-slate-600 dark:bg-slate-900/50 dark:text-slate-300">
{`YouTube live chat ──► Collector (yt-dlp / API / file)
                          │
                          ▼
                   chat_messages (SQLite)
                          │
                   Windowing (200 messages)
                          │
                          ▼
   ┌──────────────── NLP Pipeline ────────────────┐
   │ clean → tokenize → stopwords → features      │
   │ → topic → sentiment → keywords → summary     │
   │ → entities                                   │
   └──────────────────────┬───────────────────────┘
                          ▼
              news + nlp_analyses (SQLite)
                          │
                    FastAPI REST API
                          │
              React + Recharts dashboard`}
        </pre>
      </Section>

      <Section title="ชุดข้อมูล (Dataset)">
        {stats.data ? (
          <ul className="space-y-1 tabular-nums" lang="th">
            <li>ข้อความแชทจริงที่เก็บได้: {stats.data.total_messages.toLocaleString()} ข้อความ</li>
            <li>จากสตรีม: {stats.data.total_streams} รายการ</li>
            <li>วิเคราะห์ความรู้สึกรายข้อความ: {stats.data.scored_messages.toLocaleString()}</li>
            <li>กรองเป็นสแปม/ไม่มีเนื้อหา: {stats.data.noise_messages.toLocaleString()}</li>
            <li>เอกสารที่วิเคราะห์แล้ว: {stats.data.total_news.toLocaleString()} ({stats.data.chat_windows} ช่วงแชท, {stats.data.articles} บทความ)</li>
          </ul>
        ) : (
          <p className="text-sm text-slate-400">กำลังโหลด…</p>
        )}
        {streams.data && streams.data.length > 0 && (
          <ul className="mt-3 space-y-1 text-xs text-slate-500 dark:text-slate-400">
            {streams.data.map((stream) => (
              <li key={stream.id} className="truncate" lang="th">
                • {stream.channel} — {stream.message_count.toLocaleString()} ข้อความ
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="โมเดล (Model)">
        {backends.data ? (
          <ul className="space-y-1 text-sm" lang="th">
            {backends.data.map((row) => (
              <li key={row.stage}>
                <strong>{row.stage}</strong>: <code>{row.active}</code>{' '}
                {row.trained ? (
                  <span className="text-positive">(พร้อมใช้งาน)</span>
                ) : (
                  <span className="text-amber-600 dark:text-amber-400">
                    (baseline{row.note ? ` — ${row.note}` : ''})
                  </span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-400">กำลังโหลด…</p>
        )}
        <p className="mt-3 text-sm text-slate-500 dark:text-slate-400" lang="th">
          ทุกส่วนของ NLP ออกแบบให้สลับได้ผ่าน registry เพียงแก้ค่าใน .env
          จึงเปลี่ยนไปใช้โมเดลที่ฝึกเองหรือ Transformer ได้โดยไม่ต้องแก้โค้ดส่วนอื่น
        </p>
      </Section>

      <Section title="การประเมินผล (Evaluation)">
        <p lang="th">
          ดูหน้า <strong>ประเมินผลโมเดล</strong> สำหรับค่า Accuracy, Precision, Recall,
          F1-score และ Confusion Matrix ที่วัดจากชุดทดสอบซึ่งโมเดลไม่เคยเห็น
          พร้อมเปรียบเทียบกับโมเดลฐานแบบกฎบนชุดทดสอบเดียวกัน
        </p>
      </Section>
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card p-5">
      <h3 className="mb-3 font-semibold text-slate-900 dark:text-white">{title}</h3>
      <div className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
        {children}
      </div>
    </section>
  )
}

function Technique({ name, detail }: { name: string; detail: string }) {
  return (
    <div>
      <dt className="font-medium text-slate-700 dark:text-slate-200">{name}</dt>
      <dd className="text-slate-500 dark:text-slate-400">{detail}</dd>
    </div>
  )
}
